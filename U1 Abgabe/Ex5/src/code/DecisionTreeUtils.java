package code;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

public class DecisionTreeUtils {

    public static double entropyOnSubset(ARFFFile dataset, int[] indices, String c) {

        // make the subset with the given indices
        ARFFFile subset = makeSubset(dataset, indices);

        // split the data set for values of c
        ArrayList<ArrayList<String[]>> split = split(subset, c);

        // The probability of each value that the attribute c can take.
        double[] probabilities = new double[split.size()];

        for(int j = 0; j < probabilities.length; j++) {
            probabilities[j] = (double)split.get(j).size() / (double)subset.getFileData().size();
        }
        System.out.println("Distribution: " + Arrays.toString(probabilities));

        // calculate the entropy
        double entropy2 = 0;
        for (double d : probabilities) {
            entropy2 -= d * (Math.log(d) / Math.log(2));
        }

        System.out.println("Entropy on this subset: " + entropy2);
        return entropy2;
    }

    public static double informationGain(ARFFFile dataset, int[] indices, String c, String A) {

        double initialEntropy = entropyOnSubset(dataset, indices, c);
        System.out.println("INITIAL ENTROPY: " + initialEntropy + "\n");


        // make the subset.
        ARFFFile subset = makeSubset(dataset, indices);


        // split the data set for values of A.
        ArrayList<ArrayList<String[]>> split = split(subset,A);

        // calculate the information gain.
        double informationGain = initialEntropy;
        for (int i = 0; i < split.size(); i++) {

            // make a data set with only the instances that take value A for attribute c.
            ARFFFile tmp = new ARFFFile();
            tmp.attributes = subset.attributes;
            tmp.fileData = split.get(i);
            int[] indi = new int[split.get(i).size()];
            for(int j = 0; j < indi.length; j++){
                indi[j] = j;
            }

            informationGain -= ( (double) split.get(i).size() / subset.getFileData().size() )
                    * entropyOnSubset(tmp, indi, c);
            System.out.println("Entropy worth: " + split.get(i).size() + " / "
                    + subset.getFileData().size() + "\n");
        }

        return informationGain;
    }

    /*
        Makes a subset of the given data set, using the given indices.
     */
    private static ARFFFile makeSubset(ARFFFile dataset, int[] indices) {
        // MAKE THE SUBSET
        ARFFFile subset = new ARFFFile();

        ArrayList<String[]> data = dataset.getFileData();

        subset.attributes = dataset.attributes;
        subset.setRelation(dataset.getRelation());

        for (int i : indices) {
            subset.addData(data.get(i));
        }

        return subset;
    }

    /*
        auxiliary function that returns the index of a given attribute in a data set.
     */
    private static int getAttributeIndex (ARFFFile dataset, String c) {
        int index = 0;
        System.out.println("Looking for attribute: " + c);

        for (String[] attribute : dataset.getAttributes()) {

            if (attribute[0].equals(c)) {
                return index;
            }
            index++;
        }

        return -1;
    }

    /*
        Splits the data at all possible values for the given Attribute c.
     */
    private static ArrayList<ArrayList<String[]>> split(ARFFFile dataset, String c) {

        ArrayList<String[]> data = dataset.getFileData();

        int index = getAttributeIndex(dataset, c);

        ArrayList<String[]> possibilities = new ArrayList<>();
        HashSet<String> set = new HashSet<>();


        for (String[] row : dataset.fileData) {
            if (!set.contains(row[index])) {
                possibilities.add(row);
                set.add(row[index]);
            }
        }
        System.out.println("Number of possibilities for attribute '" + c + "': " + possibilities.size());

        ArrayList<ArrayList<String[]>> split = new ArrayList<>();
        for (int i = 0; i < possibilities.size(); i++){
            split.add(new ArrayList<String[]>());
        }

        for(String[] inst : data) {
            for (int i = 0; i < possibilities.size(); i++)
                if (inst[index].equals(possibilities.get(i)[index])) {
                    split.get(i).add(inst);
                }
        }

        return split;
    }
}