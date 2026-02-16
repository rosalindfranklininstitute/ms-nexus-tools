def print_item(item, offset=""):
    try:
        name = f"{item.name}:"
    except AttributeError:
        name = ""

    print(f"{offset} {name} {str(item)}")
    for at in item.attrs:
        print(f"{offset}@{at}: {item.attrs[at]}")


def print_group(d, offset=""):
    if "keys" not in dir(d):
        print_item(d, offset)
        return
    keys = list(d.keys())
    mx = len(keys)
    if mx < 10:
        indices = range(0, mx)
        for k in keys:
            print_item(d[k], offset)
            print_group(d[k], offset=offset + "- ")

    else:
        indices = [*range(0, 5), *range(-5, 0)]
        for ii in range(0, 5):
            k = keys[ii]

            print_item(d[k], offset)
            print_group(d[k], offset=offset + "- ")
        print(" " * len(offset) + "...")
        for ii in range(-5, 0):
            k = keys[ii]

            print_item(d[k], offset)
            print_group(d[k], offset=offset + "- ")
