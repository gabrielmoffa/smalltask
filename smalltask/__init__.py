"""smalltask: define tools and agents as code, run them anywhere."""


def tool(fn):
    """Mark a function as a smalltask tool.

    Only decorated functions are exposed to agents when @tool is used in a file.
    Files without any @tool decorators expose all public functions (backward compat).

    Usage::

        from smalltask import tool

        @tool
        def get_orders(days: int) -> list:
            \"\"\"Return all orders placed in the last N days.\"\"\"
            ...
    """
    fn._smalltask_tool = True
    return fn
