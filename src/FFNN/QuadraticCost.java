package FFNN;

/** Methods for computing the mean squared error cost function
 * @author Walter
 *
 */
public class QuadraticCost {
	/** Computes the mean squared error 
	 * @param actual the actual output
	 * @param expected the expected output
	 * @return the mean squared error
	 */
	public static double of(double[] actual, double[] expected){
		if(actual.length!= expected.length){
			System.out.println("QuadraticCost.of cant take arrays of different size");
		}
		int length = actual.length;
		double result = 0;
		double[] diff = Matrix.subt(actual, expected);
		for(int i=0; i<length; i++){
			result+= Math.pow(diff[i], 2);
		}
		result/= 2;
		return result;
	}
	
	/** Computes the derivatives of the mean squared error with respect to each of the output variables
	 * @param actual the actual output
	 * @param expected the expected output
	 * @return derivatives of the mean squared error for each output variable
	 */
	public static double[] primeOf(double[] actual, double[] expected){
		if(actual.length!= expected.length){
			System.out.println("QuadraticCost.primeOf cant take arrays of different size");
		}
		int length = actual.length;
		double[] diff = Matrix.subt(actual, expected);
		
		
		return diff;
	}
}
