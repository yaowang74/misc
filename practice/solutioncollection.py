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

    def twoSum(self, nums: List[int], target: int) -> List[int]:
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
