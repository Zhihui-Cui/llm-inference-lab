"""Assistant-authored M08–M10 acceptance checks; does not modify learning files.

Run from the repository: python -B tests/verify_m08_m10.py --output result.json
The AST instrumentation only exposes main() locals / substitutes exercise inputs.
All attention calculations under test still execute the archived learning source.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

RTOL, ATOL = 1e-5, 1e-6


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    assert torch.isfinite(actual).all()
    return float((actual - expected).abs().max().detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    folder = repo / 'learning/02_text'
    sys.path.insert(0, str(folder))
    sys.dont_write_bytecode = True
    torch.manual_seed(42)
    sources = {name: (folder / name).read_text(encoding='utf-8') for name in (
        'multihead_attention.py', 'token_position_embedding.py', 'char_vocab.py')}
    result = {
        'executor': 'assistant', 'purpose': 'code acceptance; not independent learner assessment',
        'executed_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'device': 'cpu', 'dtype': 'torch.float32', 'seed': 42,
        'rtol': RTOL, 'atol': ATOL, 'dropout_p': 0.0,
        'sources_sha256': {f'learning/02_text/{n}': hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in sources},
        'checks': {}, 'entrypoint_runs': {},
    }
    checks = result['checks']
    for name in ('multihead_attention.py', 'token_position_embedding.py'):
        run = subprocess.run([sys.executable, '-B', str(folder/name)], cwd=repo,
                             capture_output=True, text=True, timeout=60)
        result['entrypoint_runs'][name] = {'exit_code': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr}
        assert run.returncode == 0, run.stderr

    def heads(**overrides):
        tree = ast.parse(sources['multihead_attention.py'])
        seen = set()
        namespace = {'__name__': 'audit_multihead', 'torch': torch}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in overrides:
                    namespace['_override_' + name] = overrides[name]
                    node.value = ast.Name(id='_override_' + name, ctx=ast.Load())
                    seen.add(name)
        assert seen == set(overrides), 'exercise assignments changed; update instrumentation'
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.fix_missing_locations(tree), 'multihead_attention.py', 'exec'), namespace)
        return namespace

    base = heads()
    split, merge = base['split_heads'], base['merge_heads']
    shape_results = []
    for T, D, H in ((3,4,1), (3,4,2), (3,4,4), (2,6,3)):
        marked = torch.arange(1, T*D+1, dtype=torch.float32).reshape(T,D)
        parts = split(marked, H)
        assert list(parts.shape) == [H,T,D//H]
        assert torch.equal(merge(parts), marked)
        for h in range(H):
            assert torch.equal(parts[h], marked[:, h*(D//H):(h+1)*(D//H)])
        shape_results.append({'T': T, 'D': D, 'H': H, 'head_shape': list(parts.shape)})
    for H in (1,2,4):
        variant = heads(num_heads=H)
        assert list(variant['final_outputs'].shape) == [3,4]
        assert torch.isfinite(variant['final_outputs']).all()
    rejected = []
    for H in (0, -1, 3, 1.5, True):
        try:
            split(base['queries'], H)
        except ValueError:
            rejected.append(repr(H))
        else:
            raise AssertionError(f'invalid heads accepted: {H!r}')
    wrong_merge = base['head_queries'].reshape(3,4)
    assert not torch.equal(wrong_merge, base['queries'])
    checks['M08'] = {'marked_shapes': shape_results, 'invalid_heads_rejected': rejected,
                     'direct_reshape_negative_control_detected': True,
                     'output_projection_shape': list(base['final_outputs'].shape)}

    comparisons = {}
    comparisons['T3'] = close(base['head_outputs'], base['reference_outputs'])
    comparisons['T2'] = close(base['actual_short'], base['reference_outputs_short'])
    comparisons['prefix'] = close(base['actual_short'], base['head_outputs'][:,:2,:])
    batch_ref = F.scaled_dot_product_attention(base['q_batch'], base['k_batch'], base['v_batch'], base['allowed'], dropout_p=0.0)
    comparisons['batch_sdpa'] = close(base['batch_outputs'], batch_ref)
    comparisons['batch_A_single'] = close(base['batch_outputs'][0], base['head_outputs'])
    comparisons['batch_B_plus10'] = close(base['batch_outputs'][1], base['head_outputs'] + 10)

    # Distinct Q/K/V prevent equal-input batches from concealing shared weights.
    generator = torch.Generator().manual_seed(123)
    q, k, v = [torch.randn(2,2,3,2, generator=generator) for _ in range(3)]
    batch = heads(q_batch=q, k_batch=k, v_batch=v)
    reference = F.scaled_dot_product_attention(q,k,v,base['allowed'],dropout_p=0.0)
    comparisons['distinct_batch_sdpa'] = close(batch['batch_outputs'], reference)
    assert not torch.allclose(batch['weights_batch'][0], batch['weights_batch'][1])
    q_changed, k_changed, v_changed = q.clone(), k.clone(), v.clone()
    q_changed[1] += 1; k_changed[1, :, 1] -= 2; v_changed[1] += 10
    changed = heads(q_batch=q_changed, k_batch=k_changed, v_batch=v_changed)
    comparisons['batch_isolation'] = close(changed['batch_outputs'][0], batch['batch_outputs'][0])
    assert not torch.allclose(changed['batch_outputs'][1], batch['batch_outputs'][1])

    zero = heads(queries=torch.zeros_like(base['queries']))
    close(zero['head_outputs'][1,1], torch.tensor([0.,30.]))
    unmasked = F.scaled_dot_product_attention(zero['head_queries'],zero['head_keys'],zero['head_values'],dropout_p=0.0)
    close(unmasked[1,1], torch.tensor([0.,40.]))
    assert not torch.allclose(zero['head_outputs'][1,1], unmasked[1,1])
    future_v = base['values'].clone(); future_v[2] += 1000
    future = heads(queries=torch.zeros_like(base['queries']), values=future_v)
    comparisons['future_V_prefix'] = close(zero['head_outputs'][:,:2], future['head_outputs'][:,:2])
    future_unmasked = F.scaled_dot_product_attention(future['head_queries'],future['head_keys'],future['head_values'],dropout_p=0.0)
    assert not torch.allclose(unmasked[:,:2], future_unmasked[:,:2])
    single = heads(queries=base['queries'][:1], keys=base['keys'][:1], values=base['values'][:1])
    comparisons['T1_sdpa'] = close(single['head_outputs'], single['reference_outputs'])
    close(single['head_outputs'], single['head_values'])
    checks['M09'] = {'max_abs_errors': comparisons, 'batch_shape': list(batch['batch_outputs'].shape),
                     'mask_true_means_allowed': True, 'masked_query2_head2': zero['head_outputs'][1,1].tolist(),
                     'unmasked_query2_head2': unmasked[1,1].tolist(),
                     'no_mask_future_change_detected': True, 'distinct_batch_seed': 123}

    def positions(reverse=False, old_v=False):
        source = sources['token_position_embedding.py']
        if reverse:
            old = 'swapped_ids = ids.clone()\n    swapped_ids[2] = ids[0]'
            assert source.count(old) == 1
            source = source.replace(old, 'swapped_ids = ids.flip(0)')
        if old_v:
            old = 'swapped_attention_weights @ swapped_values'
            assert source.count(old) == 1
            source = source.replace(old, 'swapped_attention_weights @ values')
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == 'main']
        assert len(functions) == 1
        functions[0].body.append(ast.parse('return locals()').body[0])
        namespace = {'__name__':'audit_positions'}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.fix_missing_locations(tree), 'token_position_embedding.py','exec'), namespace)
            return namespace['main']()

    p = positions()
    assert list(p['ids'].shape) == [3]
    for name in ('token_vectors','position_vectors','x','queries','keys','values','attention_outputs'):
        assert list(p[name].shape) == [3,2]
        assert torch.isfinite(p[name]).all()
    future_error = close(p['attention_outputs'][:2],p['swapped_attention_outputs'][:2])
    assert not torch.allclose(p['attention_outputs'][2], p['swapped_attention_outputs'][2])
    rev = positions(reverse=True)
    close(rev['token_vectors'].flip(0),rev['swapped_tokens'])
    close(rev['x'][1],rev['swapped_x'][1])
    close(rev['queries'][1],rev['swapped_queries'][1])
    assert not torch.allclose(rev['x'][0],rev['swapped_x'][2])
    assert not torch.allclose(rev['attention_outputs'][1],rev['swapped_attention_outputs'][1])
    close(rev['swapped_attention_outputs'][0],rev['swapped_values'][0])
    assert not torch.allclose(rev['swapped_values'][0],rev['values'][0])
    try:
        positions(reverse=True, old_v=True)
    except AssertionError:
        old_v_caught = True
    else:
        raise AssertionError('reverse test did not reject reuse of original V')
    p['attention_outputs'].square().sum().backward()
    for name in ('embedding','position_embedding'):
        grad = p[name].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)>0
    checks['M10'] = {'ids_shape':[3], 'input_output_shape':[3,2], 'max_positions':p['max_positions'],
                    'future_change_prefix_max_abs_error':future_error,
                    'original_position2':rev['attention_outputs'][1].detach().tolist(),
                    'reversed_position2':rev['swapped_attention_outputs'][1].detach().tolist(),
                    'same_character_embedding_and_same_middle_query_verified':True,
                    'same_character_moved_changes_input':True,
                    'old_V_negative_control_detected':old_v_caught,
                    'both_embedding_tables_have_finite_nonzero_gradients':True,
                    'optimizer_update_performed':False}
    result['status'] = 'passed'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'checks':checks},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
