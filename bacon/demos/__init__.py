"""Built-in BACON demos that ship with the package.

Each demo is a module exposing a ``run(args)`` function.
The registry maps CLI names to (module_path, short_description) pairs.
"""

REGISTRY = {
    'hello-world': (
        'bacon.demos.hello_world',
        'Discover a random Boolean expression from synthetic data',
    ),
}
