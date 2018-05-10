HOW TO RUN THE PROGRAM:
	(option 1)
		To run this programm, open a command line in this directory and execute the following statements:
		>	javac src/code/*.java
		>	java -classpath src code/Main

		This will run the exercise on src/res/weather.nominal.arff,
		with the test values: 
		C = "play";
		A = "windy";
		indices = {0,1,2,3};

	OR

	(option 2) (Windows user)
		Double click the 'run.bat' file.




HOW TO USE DIFFERENT PARAMETERS:

	1.
	how to use a different .arff file:
		place the file inside the 'src/res' folder.


	2.
	To use other parameters then the default ones, you have to specify command line input parameters.

	To override the default values, instead run the file with:
	>	javac src/code/*.java
	>	java -classpath src code/Main FILE_NAME INDICES C A

	Example:
	>	javac src/code/*.java
	>	java -classpath src code/Main weather.nominal.arff 0,1,2,3,4,5,6,7 play windy  
	(Use this exact semantics.)

	Alternatively you can change the default values manually in Main.java




PROJECT STRUCTURE:
	All .java files are located in the '/src/code' directory.
	Exercise a) and b) are realized in decisionTreeUtilities.java.
	Expercise c) is realized in ARFFFile.java

	Main.java holds the main method for the programm.

	.ARFF files must be located in the '/src/res' directory.