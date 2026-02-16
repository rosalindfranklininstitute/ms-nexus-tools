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
    mx = len(d.keys())
    if mx < 10:
        for k in d.keys():
            print_item(d[k], offset)
            print_group(d[k], offset=offset + "- ")

    else:
        for ii in range(0, 5):
            k = d.keys()[ii]

            print_item(d[k], offset)
            print_group(d[k], offset=offset + "- ")
        print(" " * len(offset) + "...")
        for ii in range(-5, 0):
            k = d.keys()[ii]

            print_item(d[k], offset)
            print_group(d[k], offset=offset + "- ")
