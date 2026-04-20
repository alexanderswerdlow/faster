import collections
import collections.abc

for _name in dir(collections.abc):
    if _name.startswith("_"):
        continue
    if hasattr(collections, _name):
        continue
    setattr(collections, _name, getattr(collections.abc, _name))
