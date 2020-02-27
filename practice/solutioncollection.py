class SolutionCollection:
    """
    Personel practice on python

    """

    """
    Two Sum
    Problem link:
        https://leetcode.com/problems/two-sum/

    Difficulty(Easy, Medium, Hard): Easy

    Description:
        Given an array of integers, return indices of the two numbers such
        that
        they add up to a specific target.

        You may assume that each input would have exactly one solution,
        and you
        may not use the same element twice.

        Example:

        Given nums = [2, 7, 11, 15], target = 9,

        Because nums[0] + nums[1] = 2 + 7 = 9,
        return [0, 1].

    Original Solution link:
        https://codesays.com/2014/solution-to-two-sum-by-leetcode/

    Is new solution?(Yes/No): No

    New solution date(if new solution):
    """

    def twoSum(self, nums: list, target: int):
        """
        Parameters
        ---------------
        nums: List. List of integers, that each value is dinstinct
        target: int. Target integer.

        Returns
        ----------------
        List of indices of nums that the sum of two elements equals target.
        """
        mapping = dict()

        for i, e in enumerate(nums):
            complement = target - e
            if complement in mapping:
                j = mapping.get(complement)
                if j != i:
                    return [j, i]

            mapping[e] = i

    def stringPermutation(self, string: str, start: int, end: int):
        """
        Output all permutations of a given string.

        Parameters
        ------------
        string: str. Input string
        start: int. Start position of the string
        end: int. End position of the string

        Returns
        ------------
        list. List of all permutation of a string

        """
        assert start >= 0 and end <= len(string)

        def toString(lst: list):
            """

            Parameters
            ----------
            List: list.

            Returns
            -------

            """
            return ''.join(lst)

        if start == end:
            print(toString(list(string)))

        else:
            for i in range(start, end+1):
                string[start], string[i] = string[i], string[start]
                self.stringPermutation(string, start+1, end)

                string[start], string[i] = string[i], string[start]

