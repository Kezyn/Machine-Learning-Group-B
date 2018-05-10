package code;

import java.io.*;
import java.util.Arrays;


public class Main {

    public static String FILE_PATH = "src\\res\\weather.nominal.arff";
    public static String C = "play";
    public static String A = "windy";
    public static int[] indices = new int[]{0,1,2,3};

    public static void main(String[] args) {

        if(args.length == 4) {
            FILE_PATH = "src\\res\\" + args[0];
            indices = makeIntArray(args[1]);
            C = args[2];
            A = args[3];
        }


        File file = new File(FILE_PATH);

        ARFFFile arff = ARFFFile.parseARFF(file);


        System.out.println("Calculating entropy: ");
        double entropy = DecisionTreeUtils.entropyOnSubset(arff, indices, C);
        System.out.println("Calculating information gain: ");
        double informationGain = DecisionTreeUtils.informationGain(arff, indices, C, A);

        System.out.println("\n");
        System.out.println("-------------------");
        System.out.println("\n");
        System.out.println("Using subset " + Arrays.toString(indices) + " of '" + file.getName() + "'");
        System.out.println("Entropy on class attribute '" + C + "': " + entropy);
        System.out.println("Information gain on class attribute '" + C + "' when splitting on attribute '" +
                A + ": " + informationGain);
        System.out.println();
    }

    public static int[] makeIntArray(String s) {
        String[] numberStrs = s.split(",");
        int[] numbers = new int[numberStrs.length];
        for(int i = 0;i < numberStrs.length;i++) {
            numbers[i] = Integer.parseInt(numberStrs[i]);
        }
        return numbers;
    }
}