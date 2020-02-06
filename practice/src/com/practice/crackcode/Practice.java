package com.practice.crackcode;

import java.util.HashMap;
import java.util.Map;

public class Practice {
    public static void main(String[] args) {
        System.out.println("This is JAVA practice.");
        int[] nums = new int[] {1,2,3,4,5,6,7};
        int target = 9;
        int[] results = new int[7];

        results = twoSum(nums, target);
        System.out.println(results);
    }

    public static int[] twoSum(int[] nums, int target) {
        /*
         * Assumptions:
         * 1. each input would have exactly one solution
         * 2. you may not use same element twice
         * 3. sorted in ascending order
         * */
        if (nums == null || nums.length < 2) {
            return new int[0];
        }
//        initialize the result
        int[] result = new int[0];
        Map<Integer, Integer> map = new HashMap<>();
//        traverse the array
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                result = new int[2];
                result[0] = map.get(target - nums[i]);
                result[1] = i;
                return result;
            }
            map.put(nums[i], i);
        }
        return result;
    }
}
