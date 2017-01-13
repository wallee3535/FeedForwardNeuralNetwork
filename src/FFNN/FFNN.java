package FFNN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;



/** Based off of Michael Nielsen's python code at: 
 * https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
 * Feed forward neural network example using backpropagation and stochastic gradient descent,
 * created as a learning exercise.
 * Created by Walter on 12/21/2016.
 */

public class FFNN {
	//number of layers in the network, including input, hidden, and output layers
    private int numLayers;
    //sizes of the input layers
    private int[] sizes;
    //biases[l][n] is the bias of the n-th neuron on the l-th layer 
    private double[][] biases;
    //gradientBiases[l][n] is the partial derivative of the cost function with respect
    //to the the bias of the n-th neuron on the l-th layer
    private double[][] gradientBiases;
    //error[l][n] is the partial derivative of the cost function with respect to the weighted
    //input at the n-th neuron on the l-th layer
    private double[][] errors;
    //activations[l][n] is the activation at the n-th neuron on the l-th layer
    private double[][]activations;
    //weightedInputs[l][n] is the weighted input at the n-th neuron on the l-th layer
    private double[][] weightedInputs;
    //weights[l][j][k] is the weight from the k-th neuron in the l-th layer to the j-th 
    //neuron in the (l+1)-th layer
    private double[][][] weights;
    //gradientWeights[l][j][k] is the partial derivative of the cost function with respect to 
    //the weight from the k-th neuron in the l-th layer to the j-th neuron in the (l+1)-th layer
    private double[][][] gradientWeights;

    /** Constructs a neural network given the sizes of each layer
     * @param sizes sizes for each neural network layer
     */
    public FFNN(int[] sizes){
    	if(sizes.length<2){
    		System.out.println("FFNN: length of sizes must be at least 2");
    		return;
    	}
    	
    	//save sizes information
        numLayers = sizes.length;
        this.sizes = sizes;

        //initialize arrays to correct lengths
        biases = new double[numLayers-1][];
        gradientBiases = new double[numLayers-1][];
        errors = new double[numLayers-1][];
        weightedInputs = new double[numLayers-1][];
        
        for(int i =1; i< numLayers; i++){
            int y = sizes[i];
            biases[i-1] = new double[y];
            gradientBiases[i-1]= new double[y];
            errors[i-1] = new double[y];
            weightedInputs[i-1] = new double[y];
            
        }

        activations = new double[numLayers][];
        for(int i=0; i<numLayers; i++){
        	int y = sizes[i];
        	activations[i] = new double[y];
        }
        
        weights = new double[numLayers-1][][];
        gradientWeights = new double[numLayers-1][][];
        for(int i = 0,j=1; i<numLayers-1; i++, j++){
            int x = sizes[i];
            int y = sizes[j];
            weights[i] = new double[y][x];
            gradientWeights[i] = new double[y][x];
        }
        
        //fill weights and biases with random Gaussian doubles
        Matrix.gaussian(weights);
        Matrix.gaussian(biases);

    }

    /**Computes neural network output for the input data
     * @param input input for neural network
     * @return result of neural network computation
     */
    public double[] feedforward(double[] input){
        if(input.length != sizes[0]){
            System.out.println("FFNN:feedforward: given array size does not match size of first layer");
            return null;
        }
        for(int i = 0; i< weights.length; i++){
            input = Matrix.multT(weights[i], input);
            input = Matrix.add(input, biases[i]);
            input = Sigmoid.of(input);
        }
        return input;
    }

    /** Trains the neural network with stochastic gradient descent
     * @param trainingData trainingData[i][0] and trainingData[i][1] are the inputs and outputs, 
     * respectively for the i-th data point to train the neural network with
     * @param epochs number of epochs (iterations) to train the network for
     * @param miniBatchSize number of data points to check before updating neural network 
     * @param eta learning rate
     * @param testData testData[i][0] and testData[i][1] are the inputs and outputs, 
     * respectively for the i-th data point to test the neural network with. Can be null.
     * 
     */
    public void SGD(double[][][] trainingData, int epochs, int miniBatchSize, double eta, 
    		double[][][] testData){
    	//
    	for(int e = 0; e< epochs; e++){
    		//Ramdomly shuffle the training data
    		Collections.shuffle(Arrays.asList(trainingData));
    		//counter for each data point
    		int k = 0;
    		ArrayList<double[][]> miniBatch = new ArrayList<double[][]>();
    		do{
    			//if enought datapoints are in  miniBatch, update the network with the miniBatch
    			if(miniBatch.size()!= 0 && k%miniBatchSize==0){
    				updateMiniBatch((miniBatch.toArray(new double[miniBatchSize][][])), eta);
    				miniBatch.clear();
    				//if test data is given, compute and print the cost after each mini-batch
    				if(testData!=null){
    					double cost = evaluateTestData(trainingData);
    					System.out.println("Cost: "+cost);
    				}
    			}
    			//add the current data point to trainingData
    			miniBatch.add(trainingData[k]);
    			k++;
    		}while(k<trainingData.length);
    		//add last update
    		updateMiniBatch((miniBatch.toArray(new double[miniBatchSize][][])), eta);
			miniBatch.clear();
			if(testData!=null){
				double cost = evaluateTestData(trainingData);
				System.out.println("Cost: "+cost);
			}
    	}
    }
    
    /** Updates the network with a minibatch of data points
     * @param miniBatch miniBatch[i][0] and miniBatch[i][1] are the inputs and outputs, 
     * respectively for the i-th data point to train the neural network with
     * @param eta learning rate
     */
    private void updateMiniBatch(double[][][] miniBatch, double eta){
    	//clear gradient arrays, to update later in backprop
    	Matrix.clear(gradientWeights);
    	Matrix.clear(gradientBiases);
    	for(int d = 0; d<miniBatch.length; d++){
    		//back propagate and update gradientWeights and gradientBiases
    		backprop(miniBatch[d][0], miniBatch[d][1]);
    	}
    	//adjust weights using gradientWeights
    	for(int l= 0; l < weights.length; l++){
    		double[][] mat = weights[l];
    		for( int r = 0 ; r< mat.length; r++){
    			for(int c=0; c< mat[0].length; c++){
    				weights[l][r][c]-=eta * gradientWeights[l][r][c];
    			}
    		}
    	}
    	//adjust biases using gradientBiases
    	biases = Matrix.subt(
    		biases, 
    		Matrix.scale(gradientBiases, eta)
    	);
    	
    }
    
    /** Backpropagation adding the gradientWeights and gradientBiases
     * @param input input for a single data point
     * @param desiredOutput output for the data point
     */
    private void backprop(double[] input, double[] desiredOutput){
    	if(input.length!= sizes[0]){
    		System.out.println("FFNN.backprop: input does not match size of first layer");
    		return;
    	}
    	if(desiredOutput.length!= sizes[numLayers-1]){
    		System.out.println("FFNN.backprop: desiredOutput does not match size of last layer");
    		return;
    	}
    	
    	//store weighted inputs and activations during forward propagoation
    	double[] a = input;
    	activations[0] = a;
    	for(int i = 0; i< weights.length; i++){
            a = Matrix.multT(weights[i], a);
            a = Matrix.add(a, biases[i]);
            weightedInputs[i]=a;
            a = Sigmoid.of(a);
            activations[i+1]=a;
        }
    	//get error of the last layer
    	errors[numLayers-2] = Matrix.hadamark(
    			QuadraticCost.primeOf(activations[numLayers-1], desiredOutput),
    			Sigmoid.primeOf(weightedInputs[numLayers-2]) 
    			);
    	//recursively get error for previous layer through back propagation
    	for(int i = numLayers-3; i>=0; i--){
    		double[] dot = Matrix.multT(Matrix.transpose(weights[i+1]), errors[i+1]);
    		double[] sigPrime = Sigmoid.primeOf(weightedInputs[i]);
    		errors[i] = Matrix.hadamark(dot, sigPrime);
    	}
    	//get weight and bias gradients
    	for(int l=0; l<numLayers-1; l++){
    		double[] curBias = errors[l].clone();
    		gradientBiases[l]= Matrix.add(gradientBiases[l], curBias);
    		double[][] curWeight = Matrix.mult(
    				Matrix.colToMat(errors[l].clone()), 
    				Matrix.rowToMat(activations[l])
    				);
    		gradientWeights[l]= Matrix.add(gradientWeights[l], curWeight);
    				
    	}
    	
    	
    }
    
    /** Evaluate test data with the neural network and return mean squared error
     * @param testData testData[i][0] and testData[i][1] are the inputs and outputs, 
     * respectively for the i-th data point to test the neural network with.
     * @return mean squared error for data point
     */
    public double evaluateTestData(double[][][] testData){
		double[] actual;
		double[] expected;
		double totalError = 0;
		for(int i = 0; i<testData.length; i++){
			actual =feedforward(testData[i][0]);
			expected = testData[i][1];
			//accumulate error for each test data point
			totalError+=QuadraticCost.of(actual, expected);
		}
		//return average error
		return totalError/testData.length;
		
	}
    




}
