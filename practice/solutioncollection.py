class SolutionCollection:
    """Personel practice on python

    """

    """ Two Sum
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

    def twoSum(self, num, target):
        """

        :param num: input array of number
        :param target: target that sum of two elements from num must equal
        :return: tuple of indecies of two number in num that their sum is
        target
        """
        mapping = dict()
        for i in range(len(num)):
            x = num[i]
            if target-x in mapping:
                return [mapping[target-x], i]
            mapping[x] = i
            # just for illustration
            print(mapping)
