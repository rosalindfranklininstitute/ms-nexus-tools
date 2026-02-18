import argparse
import csv
from typing import NamedTuple

from icecream import ic


class Isotope(NamedTuple):
    nominal: int
    accurate: float
    abundance: float


class Element(NamedTuple):
    name: str
    symbol: str
    isotopes: list[Isotope]


def element_to_str(element: Element):
    isotope_strings = [f"Isotope({i[0]},{i[1]},{i[2]})" for i in element.isotopes]
    return (
        f'Element("{element.name}", "{element.symbol}", [{",".join(isotope_strings)}])'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the SIS CSV file into a map.")
    parser.add_argument(
        "input",
        help="The input csv file. Columns should be: Name, Symbol, Nominal number, Accurate Mass, Abundance percentage",
    )
    parser.add_argument(
        "output",
        help="The output python file that will contain a map of the data.",
    )
    parser.add_argument(
        "electron_mass",
        help="The mass of an electron in Da",
    )

    args = parser.parse_args()

    data: dict[str, Element] = {}

    data["E"] = Element("Electron", "E", [Isotope(0, args.electron_mass, 100)])

    with open(args.input, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            name, symbol, nominal, accurate, abundance = row[0:5]
            name = name.strip().replace('"', "")
            symbol = symbol.strip().replace('"', "")
            nominal = int(nominal)
            accurate = float(accurate)
            abundance = float(abundance)
            if symbol in data:
                data[symbol].isotopes.append(Isotope(nominal, accurate, abundance))
            else:
                data[symbol] = Element(
                    name, symbol, [Isotope(nominal, accurate, abundance)]
                )

    with open(args.output, "w") as out_file:
        out_file.write("""
from typing import NamedTuple

class Isotope(NamedTuple):
    nominal: int
    accurate: float
    abundance: float

class Element(NamedTuple):
    name: str
    symbol: str
    isotopes: list[Isotope]

elements = {
""")
        for key, element in data.items():
            out_file.write(f'    "{key}": {element_to_str(element)},\n')
        out_file.write("}\n")
