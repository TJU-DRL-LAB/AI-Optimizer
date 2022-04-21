class Wrapper(object):
    """
    Mixin for deferring attributes to a wrapped, inner object.
    """

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, attr):
        """
        Dispatch attributes by their status as magic, members, or missing.
        - magic is handled by the standard getattr
        - existing attributes are returned
        - missing attributes are deferred to the inner object.
        """
        # don't make magic any more magical
        is_magic = attr.startswith('__') and attr.endswith('__')
        if is_magic:
            return super().__getattr__(attr)
        try:
            # try to return the attribute...
            return self.__dict__[attr]
        except:
            # ...and defer to the inner dataset if it's not here
            return getattr(self.inner, attr)

