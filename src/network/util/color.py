# Always end with ENDC
ANSI_COLORS = {
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKCYAN": "\033[96m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}


def get(selected_color: str) -> str:
    color = ANSI_COLORS.get(selected_color)
    print(color)
    if color:
        return color
    else:
        return ""


def prBold(s):
    print("\33[1m {}\33[00m".format(s))


def prUnderline(s):
    print("\33[4m {}\33[00m".format(s))


def prFail(s):
    print("\033[91m {}\033[00m".format(s))


def prOkG(s):
    print("\033[92m {}\033[00m".format(s))


def prWarning(s):
    print("\033[93m {}\033[00m".format(s))


def prOkB(s):
    print("\033[94m {}\033[00m".format(s))


def prHeader(s):
    print("\033[95m {}\033[00m".format(s))


def prOkC(s):
    print("\033[96m {}\033[00m".format(s))


def prInfo(s):
    print("\033[97m {}\033[00m".format(s))


def prBlack(s):
    print("\033[90m {}\033[00m".format(s))
