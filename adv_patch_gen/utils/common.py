"""
Common utils
"""

class BColors:
    """
    Border Color values for pretty printing in terminal
    Sample Use:
        print(f"{BColors.WARNING}Warning: Information.{BColors.ENDC}"
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
