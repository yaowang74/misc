from config import PROBLEM_DICT
from solutioncollection import SolutionCollection


def main():
    """practice entrypoint

    :return: return from individual problem
    """
    for key, val in PROBLEM_DICT.items():
        print("{}: {}".format(key, val))
    select = input("Please select a problem: ")

    try:
        select = int(select)
    except:
        raise ValueError("Please input an integer.")

    while select < 1 or select > len(PROBLEM_DICT.keys()):
        print("Invalid selection")
        select = input("Please select a problem: ")

    lc = SolutionCollection()

    if select == 1:
        num = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        print("Sample number list: {}".format(num))
        target  = input("Input a positive integer as target: ")
        try:
            target  = int(target)
        except:
            raise ValueError("Invalid input.")
        print(lc.twoSum(num=num, target=target))



if __name__ == "__main__":
    main()

