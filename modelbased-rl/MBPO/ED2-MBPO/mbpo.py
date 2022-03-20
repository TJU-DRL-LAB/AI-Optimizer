from examples.instrument import run_example_local
import sys

if __name__ == '__main__':
    model_name = 'examples.development'
    run_example_local(model_name, tuple(sys.argv[1:]))