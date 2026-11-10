import argparse
from pathlib import Path
import os
import sys


EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS / 'src'))


def main():
    parser = argparse.ArgumentParser(description='LoopJev paper experiment entry point')
    subparsers = parser.add_subparsers(dest='command', required=True)
    audit = subparsers.add_parser('audit-data')
    audit.add_argument('--data-root', default='data/Open-Jev-v1.1')
    audit.add_argument('--output', required=True)
    allocate = subparsers.add_parser('draft-data')
    allocate.add_argument('--data-root', default='data/Open-Jev-v1.1')
    allocate.add_argument('--audit-root', default='experiments/outputs/data/audit')
    allocate.add_argument('--output', required=True)
    allocate.add_argument('--max-input-tokens', type=int, default=2048)
    subparsers.add_parser('test-data')
    subparsers.add_parser('test')
    dashboard = subparsers.add_parser('tensorboard')
    dashboard.add_argument('--logdir', default='experiments/outputs/training')
    dashboard.add_argument('--port', type=int, default=6006)
    preflight = subparsers.add_parser('preflight')
    preflight.add_argument('--config', default='experiments/configs/loopjev_uniform.json')
    preflight.add_argument('--data-lock', default='experiments/protocols/data.json')
    preflight.add_argument('--per-kind', type=int, default=2)
    train = subparsers.add_parser('train')
    train.add_argument('--config', required=True)
    train.add_argument('--data-lock', default='experiments/protocols/data.json')
    train.add_argument('--output', required=True)
    train.add_argument('--smoke', action='store_true')
    train.add_argument('--smoke-per-kind', type=int, default=1)
    train.add_argument('--resume', action='store_true')
    train.add_argument('--smoke-updates', type=int, choices=[1, 2], default=1)
    train.add_argument('--stop-after-updates', type=int)
    benchmark = subparsers.add_parser('benchmark')
    benchmark.add_argument('--config', required=True)
    benchmark.add_argument('--data-lock', default='experiments/protocols/data.json')
    benchmark.add_argument('--output', required=True)
    benchmark.add_argument('--records', type=int, default=256)
    benchmark.add_argument('--warmup', type=int, default=3)
    benchmark.add_argument('--updates', type=int, default=30)
    benchmark.add_argument('--profile', action='store_true')
    benchmark.add_argument('--variant', choices=['current', 'serial', 'cached', 'budget8k', 'budget16k', 'budget32k', 'no-checkpoint',
        'causal8k', 'causal16k', 'causal32k', 'causal-no-checkpoint'], default='current')
    equivalence = subparsers.add_parser('verify-batching')
    equivalence.add_argument('--config', required=True)
    equivalence.add_argument('--data-lock', default='experiments/protocols/data.json')
    equivalence.add_argument('--output', required=True)
    equivalence.add_argument('--fp32', action='store_true')
    equivalence.add_argument('--causal-fastpath', action='store_true')
    evaluate = subparsers.add_parser('evaluate')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--data-lock', default='experiments/protocols/data.json')
    evaluate.add_argument('--role', choices=['stage1', 'risk_train', 'risk_val', 'policy_fit', 'validation'], default='validation')
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--temperature-file')
    evaluate.add_argument('--smoke', action='store_true')
    evaluate.add_argument('--smoke-per-kind', type=int, default=1)
    freeze = subparsers.add_parser('stage2-freeze')
    freeze.add_argument('--checkpoint', required=True)
    freeze.add_argument('--data-lock', default='experiments/protocols/data.json')
    freeze.add_argument('--output', required=True)
    freeze.add_argument('--frozen-model', required=True)
    extract = subparsers.add_parser('stage2-extract')
    extract.add_argument('--frozen-model', default='experiments/protocols/stage2-uniform8561.json')
    extract.add_argument('--data-lock', default='experiments/protocols/data.json')
    extract.add_argument('--role', choices=['risk_train', 'risk_val', 'policy_fit', 'validation'], required=True)
    extract.add_argument('--output', required=True)
    extract.add_argument('--smoke', action='store_true')
    extract.add_argument('--smoke-per-kind', type=int, default=1)
    risk = subparsers.add_parser('stage2-fit')
    risk.add_argument('--cache-root', required=True)
    risk.add_argument('--output', required=True)
    risk.add_argument('--seed', type=int, default=0)
    risk.add_argument('--bootstrap-repeats', type=int, default=2000)
    replay = subparsers.add_parser('stage2-validate')
    replay.add_argument('--cache', required=True)
    replay.add_argument('--risk-run', required=True)
    replay.add_argument('--output', required=True)
    replay.add_argument('--seed', type=int, default=0)
    replay.add_argument('--bootstrap-repeats', type=int, default=2000)
    dynamic = subparsers.add_parser('stage2-dynamic')
    dynamic.add_argument('--frozen-model', default='experiments/protocols/stage2-uniform8561.json')
    dynamic.add_argument('--data-lock', default='experiments/protocols/data.json')
    dynamic.add_argument('--risk-run', required=True)
    dynamic.add_argument('--output', required=True)
    dynamic.add_argument('--dynamic-records', type=int, default=128)
    dynamic.add_argument('--timing-repeats', type=int, default=3)
    dynamic.add_argument('--smoke', action='store_true')
    dynamic.add_argument('--smoke-per-kind', type=int, default=1)
    pipeline = subparsers.add_parser('stage2-run')
    pipeline.add_argument('--resume', action='store_true')
    final_lock = subparsers.add_parser('final-lock')
    final_lock.add_argument('--output', required=True)
    final_lock.add_argument('--data-lock', default='experiments/protocols/data.json')
    final_lock.add_argument('--jevbench', required=True)
    final_run = subparsers.add_parser('final-run')
    final_run.add_argument('--output', required=True)
    final_extract = subparsers.add_parser('final-extract')
    final_extract.add_argument('--output', required=True)
    final_extract.add_argument('--model', choices=['loopjev_uniform', 'loopjev_last', 'qwen_jev'], required=True)
    final_extract.add_argument('--role', choices=['calibration', 'test', 'jevbench'], required=True)
    args = parser.parse_args()
    os.chdir(EXPERIMENTS.parent)
    if args.command == 'audit-data':
        from loopjev.data.audit import audit_data
        audit_data(Path(args.data_root), Path(args.output))
    elif args.command == 'draft-data':
        from loopjev.data.allocation import allocate_data
        allocate_data(Path(args.data_root), Path(args.audit_root), Path(args.output), args.max_input_tokens)
    elif args.command == 'test-data':
        import unittest
        suite = unittest.defaultTestLoader.discover(str(EXPERIMENTS / 'tests'), pattern='test_data.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    elif args.command == 'test':
        import unittest
        suite = unittest.defaultTestLoader.discover(str(EXPERIMENTS / 'tests'))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    elif args.command == 'tensorboard':
        from tensorboard import program
        dashboard = program.TensorBoard()
        dashboard.configure(argv=['tensorboard', '--logdir', str(Path(args.logdir).resolve()), '--host', '127.0.0.1', '--port', str(args.port)])
        raise SystemExit(dashboard.main())
    elif args.command in ('benchmark', 'verify-batching'):
        from loopjev.training import performance
        getattr(performance, args.command.replace('-', '_'))(args)
    elif args.command == 'evaluate':
        from loopjev.evaluation.engine import evaluate
        evaluate(args)
    elif args.command == 'stage2-freeze':
        from loopjev.stage2.freeze import freeze
        freeze(args)
    elif args.command == 'stage2-extract':
        from loopjev.stage2.extract import extract
        extract(args)
    elif args.command == 'stage2-fit':
        from loopjev.stage2.risk import fit
        print(fit(args), flush=True)
    elif args.command == 'stage2-validate':
        from loopjev.stage2.risk import evaluate_cache
        print(evaluate_cache(args.cache, args.risk_run, args.output,
            seed=args.seed, bootstrap_repeats=args.bootstrap_repeats), flush=True)
    elif args.command == 'stage2-dynamic':
        from loopjev.stage2.dynamic import evaluate
        evaluate(args)
    elif args.command == 'stage2-run':
        import runpy
        runpy.run_path(str(EXPERIMENTS / 'scripts/run_stage2.py'))['main'](resume=args.resume)
    elif args.command in ('final-lock', 'final-run', 'final-extract'):
        from loopjev.evaluation import final
        getattr(final, {'final-lock': 'freeze', 'final-run': 'run', 'final-extract': 'extract'}[args.command])(args)
    else:
        from loopjev.training import engine
        getattr(engine, args.command)(args)


if __name__ == '__main__':
    main()
