from value import Value
from graph import plot_graph


def main():
    a = Value(1.0, label="a")
    b = Value(2.0, label="b")
    c = Value(-2.0, label="c")
    d = Value(19.0, label="d")

    e = a + b
    e.label = "e"
    f = e * c + d
    f.label = "f"

    f.backward()
    plot_graph(f)


if __name__ == "__main__":
    main()
