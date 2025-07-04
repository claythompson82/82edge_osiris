# Mutation Strategy Cookbook

<!-- markdownlint-disable MD013 -->

The DGM kernel self-modifies by generating patches to its own source files. Every
patch is produced by a *mutation strategy* – a small transformation that takes
existing code and returns a mutated version. Strategies live under
`dgm_kernel.mutation_strategies` and expose a `mutate(code: str) -> str` method.
The active strategy is selected at runtime via the `DGM_MUTATION` environment
variable.

## Built-in Strategies

| Strategy | Description | Pros | Cons |
| -------- | ----------- | ---- | ---- |
| `ASTInsertComment` | Adds a string literal at the start of the module. | Safe, rarely fails tests. | No semantic effect. |
| `ASTRenameIdentifier` | Appends `_renamed` to the first function name. | Shows reload mechanics and visible diff. | May break callers. |
| `ASTInsertCommentAndRename` | Runs both edits above when escalation is needed. | Larger diff that might yield higher reward. | More disruptive overall. |

The source file describes the two primary strategies like so:

```python
"""Mutation strategies for Darwin-Gödel Machine (DGM).

This module implements two simple AST-based transformations from
DGM design PDF § 2.3:

- ``ASTInsertComment`` inserts a no-op string literal at the start of a
  module, effectively acting as a comment.
- ``ASTRenameIdentifier`` renames the first function definition by
  appending ``_renamed`` to its name.

Both strategies parse the input code and return syntactically valid
Python source. Additional strategies can be plugged in via the
``DGM_MUTATION`` environment variable.
"""
```

### Example Patches

Given an input file `t.py` containing

```python
def greet(name):
    return f"hi {name}"
```

running the kernel with different strategies yields the following JSON patches:

#### ASTInsertComment

```json
{
  "target": "t.py",
  "before": "def greet(name):\n    return f\"hi {name}\"\n",
  "after": "\"mutated\"\n\ndef greet(name):\n    return f\"hi {name}\"\n"
}
```

#### ASTRenameIdentifier

```json
{
  "target": "t.py",
  "before": "def greet(name):\n    return f\"hi {name}\"\n",
  "after": "def greet_renamed(name):\n    return f\"hi {name}\"\n"
}
```

#### ASTInsertCommentAndRename

```json
{
  "target": "t.py",
  "before": "def greet(name):\n    return f\"hi {name}\"\n",
  "after": "\"mutated\"\n\ndef greet_renamed(name):\n    return f\"hi {name}\"\n"
}
```

## Registering Custom Strategies

`meta_loop.py` reads the strategy name from `DGM_MUTATION` and instantiates the
matching class:

```python
_MUTATION_NAME = os.getenv("DGM_MUTATION", "ASTInsertComment")
...
class ASTInsertCommentAndRename:
    def mutate(self, code: str) -> str:
        code = ASTInsertComment().mutate(code)
        return ASTRenameIdentifier().mutate(code)
```

To add your own mutation:

1. Define a class with a `mutate` method in `dgm_kernel/mutation_strategies.py`.
2. Set `DGM_MUTATION` to the class name before starting the kernel:

   ```bash
   export DGM_MUTATION=MyStrategy
   python -m dgm_kernel.meta_loop
   ```

3. The scheduler or environment variable will load your class and apply it to
   generated patches.
