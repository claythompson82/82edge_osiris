"""Match/case statement."""


def classify(val):
    match val:
        case 0:
            return "zero"
        case 1 | 2:
            return "one or two"
