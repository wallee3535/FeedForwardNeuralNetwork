package FFNN;

/** Methods for computing sigmoid function
 * @author Walter
 *
 */
public class Sigmoid {
	
	/** Sigmoid function takes a double and returns a double between 0 and 1
	 * @param x input for sigmoid function
	 * @return sigmoid of x, a double between 0 and 1
	 */
	public static double of(double x){
		return 1/(1+Math.exp(-x));
	}
	
	/** Derivative of the sigmoid function at a particular value
	 * @param x value to calculate derivative at
	 * @return derivative of sigmoid at x
	 */
	public static double primeOf(double x){
		return of(x)*(1-of(x));
	}
	
	/** Applies sigmoid to every element of vector
	 * @param vector vector to apply sigmoid to
	 * @return new vector with sigmoided values
	 */
	public static double[] of(double[] vector){
		double[] result = new double[vector.length];
		for(int i = 0 ; i< vector.length; i++){
			result[i] = of(vector[i]);
		}
		return result;
	}
	
	/** Applies sigmoid derivative to every element of a matrix
	 * @param vector vector to apply sigmoid derivative to
	 * @return new vector with sigmoided values
	 */
	public static double[] primeOf(double[] vector){
		double[] result = new double[vector.length];
		for(int i = 0 ; i< vector.length; i++){
			result[i] = primeOf(vector[i]);
		}
		return result;
	}
}
