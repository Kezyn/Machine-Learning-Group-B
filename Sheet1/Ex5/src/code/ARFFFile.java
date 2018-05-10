package code;

import java.io.*;
import java.util.ArrayList;
/**
 * A class that handles the tables for an ARFF file.
 * @author P3
 */
public class ARFFFile {

    /**
     * Name of the relation.
     */
    private String relation;

    /**
     * Name of the file.
     */
    private String fileName;

    /**
     * List of attributes with their names and types.
     */
    public ArrayList<String[]> attributes;

    /**
     * List of data values.
     */
    public ArrayList<String[]> fileData;

    /**
     * Constructor.
     */
    public ARFFFile() {
        attributes = new ArrayList<String[]>();
        fileData = new ArrayList<String[]>();
    }

    /**
     * @return Relation of the file.
     */
    public final String getRelation() {
        return relation;
    }

    /**
     * @param newRelation Relation for the file.
     */
    public final void setRelation(final String newRelation) {
        relation = newRelation;
    }

    /**
     * @return Name of the file
     */
    public final String getFileName() {
        return this.fileName;
    }

    /**
     * @param str Name of the file
     */
    public final void setFileName(final String str) {
        this.fileName = str;
    }

    /**
     * @return Number of attributes.
     */
    public final int getAttributeNumber() {
        return attributes.size();
    }

    /**
     * @return List of attributes.
     */
    public final ArrayList<String[]> getAttributes() {
        return attributes;
    }

    /**
     * @param attrName Name of the attributes representing the columns.
     * @param type Type of the attributes.
     */
    public final void addAttribute(final String attrName, final String type) {
        attributes.add(new String[]{attrName, type});
    }

    /**
     * @return List of data.
     */
    public final ArrayList<String[]> getFileData() {
        return fileData;
    }

    /**
     * @param data String array of data values.
     */
    public final void addData(final String[] data) {
        String[] newData = new String[attributes.size()];
        for (int i = 0; i < data.length; i++) {
            newData[i] = data[i];
        }
        fileData.add(newData);
    }
    //extract file name


    /*
        Basically a toString() method.
     */
    public final String buildARFF(boolean lineBreak) {
        StringBuilder sb = new StringBuilder();

        int attributeCount = getAttributeNumber();

        String newLine = " ";
        if (lineBreak) {
            newLine = "\n";
        }

        sb.append("@RELATION ");
        sb.append(getRelation());
        sb.append(newLine);
        for (String[] e : getAttributes()) {
            sb.append(newLine);
            sb.append("@ATTRIBUTE ");
            sb.append(e[0]);
            sb.append(" ");
            sb.append(e[1]);
        }


        sb.append(newLine + newLine + "@DATA");
        for (String[] e : getFileData()) {
            sb.append(newLine);
            for (int i = 0; i < attributeCount; i++) {
                sb.append(e[i]);
                if (i < attributeCount - 1) {
                    sb.append(",");
                }
            }
        }

        return sb.toString();
    }

    /*
        Makes an actual arff object from strings.
     */
    private static final ARFFFile stringToARFF(final String content, final String fileName,
                                               final String url) {
        ARFFFile arffFile = new ARFFFile();
        arffFile.setFileName(fileName);
        BufferedReader br = new BufferedReader(new StringReader(content));
        try {
            for (String s = br.readLine(); s != null; s = br.readLine()) {
                interpretLine(s, arffFile);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return arffFile;
    }

    /**
     * @param str      line to interpret
     * @param arffFile this line belongs to
     */
    private static final void interpretLine(final String str, final ARFFFile arffFile) {
        if (str.trim().isEmpty()) {
            return;
        } else if (str.trim().startsWith("%")) {
            return;
        } else if (str.trim().toLowerCase().startsWith("@attribute")) {
            attributeLine(str, arffFile);
        } else if (str.trim().toLowerCase().startsWith("@relation")) {
            relationLine(str, arffFile);
        } else if (str.trim().toLowerCase().startsWith("@data")) {
            return;
        } else {
            dataLine(str.trim(), arffFile);
        }
    }

    /**
     * @param str      line that starts with  @attribute
     * @param arffFile this attribute belongs to
     */
    private static void attributeLine(final String str, final ARFFFile arffFile) {

        /**
         * erzeugt Array von allen Strings bis zum übernächsten @(Zeilenanfang
         * nächste Zeile)
         */
        // \\s+ deletes all spaces when used in split
        String[] tempArray = str.trim().split("\\s+");
        //parameter are attributeName for 1 and attributeType 2
        arffFile.addAttribute(tempArray[1], tempArray[2]);
    }

    /**
     * @param str      line that starts with @relation
     * @param arffFile this relation belongs to
     */
    private static void relationLine(final String str, final ARFFFile arffFile) {

        String[] tempArray = str.trim().split("\\s+");
        arffFile.setRelation(tempArray[1]);
    }

    /**
     * @param str      line that comes after @data
     * @param arffFile this data belongs to
     */
    private static void dataLine(final String str, final ARFFFile arffFile) {

        String[] tempArray = str.trim().split(",");
        if (tempArray.length > 0) {

            arffFile.addData(tempArray);
        }
    }

    /**
     * @param file File (.arff) to convert to object
     * @return java respresentation of an .arff file
     */
    public static final ARFFFile parseARFF(final File file) {
        String fileName = file.getName();

        String url = file.getAbsolutePath();
        String content = contentFromArffFile(file);
        ARFFFile f = stringToARFF(content.toString(), fileName, url);
        return f;
    }

    /**
     *
     * @param arffFile
     * @return the text body of the file
     */
    private static final String contentFromArffFile(final File arffFile) {
        String content = "";
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(arffFile), "UTF-8"));
             StringWriter writer = new StringWriter();) {

            for (String s = br.readLine(); s != null; s = br.readLine()) {
                writer.write(s + "\n");
            }
            content = writer.toString();
        } catch (Exception e) {
            System.out.println("catch contentFromArffFile");
            e.printStackTrace();
        }
        return content.toString();
    }
}
