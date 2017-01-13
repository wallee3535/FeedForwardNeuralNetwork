package FFNN;

/** Runs an example training a feed forward neural network to learn the 'and' function.
 * @author Walter
 *
 */
public class Main {

	public static void main(String[] args) {
		int[] sizes = {2, 2, 1};
		FFNN net = new FFNN(sizes);
		//training data to learn the 'and' function
		double[][][] trainingData = {
				{{1, 0}, {0}},
				{{0, 1}, {0}},
				{{1, 1}, {1}},
				{{0, 0}, {0}}
		};
		//train with 100 epochs, bactch size of 4, learning rate of 1
		//also use the training data as test data
		net.SGD(trainingData, 100, 4, 1, trainingData);

		/*
		int[] bigArr = {100, 10, 1};
		double[][][] bigData = new double[1][2][];
		double[] bigInput = new double[100];
		for(int i = 0; i<100; i++) bigInput[i] = 0.5;
		bigData[0][0] = bigInput;
		bigData[0][1] = new double[1];
		bigData[0][1][0] = 0.5;
		
		FFNN bigNet = new FFNN(bigArr);
		bigNet.SGD(bigData, 100, 1, 1, bigData);
		*/
	}


}
