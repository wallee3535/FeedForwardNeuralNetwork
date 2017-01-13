package FFNN;

import java.util.Random;

/* Provides matrix operations.
 * A matrix is represented as a 2D double array, where mat[r][c] is the value of the
 * matrix mat at row r and column c. A vector is represented as a 1D array.
 * 
 */
public class Matrix {
	
	/** Perform matrix multiplication on two matrixes
	 * @param mat1 matrix to multiply on left side
	 * @param mat2 matrix to multiply on right side
	 * @return product of the two matrixes
	 */
    public static double[][] mult(double[][] mat1, double[][] mat2){
    	int rows1 = mat1.length;
    	int cols1 = mat1[0].length;
    	int rows2 = mat2.length;
    	int cols2 = mat2[0].length;
        if(cols1!= rows2){
            System.out.println("Matrix.mult() array sizes dont match");
            return null;
        }
        double[][] result = new double[rows1][cols2];
        for(int r = 0; r< rows1; r++){
            for(int c = 0; c<cols2; c++){
            	double sum = 0;
            	for(int i = 0; i< cols1; i++){
            		sum+= mat1[r][i]*mat2[i][c];
            	}
            	result[r][c]= sum;
            }
        }
        return result;
    }
    
    /** Performs matrix multiplication with mat on left side and colVec as a column vector
     * Allows column vectors to be represented as 1D arrays and repeatedly multiplied by matrixes
     * @param mat matrix to multiply on left side
     * @param colVec column vector to multiply on right side
     * @return column vector product of multiplication
     */
    public static double[] multT(double[][] mat, double[] colVec){
    	if(mat[0].length!= colVec.length){
    		System.out.println("Matrix.multT() array sizes dont match");
            return null;
    	}
    	double[] result = new double[mat.length];
    	for(int r = 0; r< mat.length; r++){
    		result[r] = dot(mat[r], colVec);
    	}
    	return result;
    }

    /** Dot product of two vectors
     * @param vec1 first vector to dot product
     * @param vec2 second vector to dot product
     * @return value of dot product
     */
    public static double dot(double[] vec1, double[] vec2){
        if(vec1.length != vec2.length){
            System.out.println("Matrix.dot(): array sizes dont match");
            return 0;
        }
        double result = 0;
        for(int i = 0 ; i< vec1.length; i++){
            result += vec1[i]*vec2[i];
        }
        return result;
    }
    
    /** Converts row vector to matrix
     * @param row row vector to convert
     * @return matrix representing vector
     */
    public static double[][] rowToMat(double[] row){
    	double[][] result = new double[1][];
    	result[0]= row;
    	return result;
    }
    
    /** Converts column vector to matrix
     * @param col column vector to convert
     * @return matrix representing vector
     */
    public static double[][] colToMat(double[] col){
    	double[][] result = new double[col.length][1];
    	for(int r= 0; r< col.length; r++){
    		result[r][0] = col[r]; 
    	}
    	return result;
    }

    /** Unary operation taking a double
     * @author Walter Ho
     */
    private interface UnaryOp{
    	public double calc(double a);
    }
    
    /** Performs a unary operation on every element in the array and returns the
     * result in a new array
     * @param vec vector to perform operation on
     * @param op operation to apply
     * @return new vector with results of binary operation
     */
    public static double[] unaryCalc(double vec[], UnaryOp op){
    	double result[] = new double[vec.length];
    	for(int i = 0; i< vec.length; i++){
    		result[i] = op.calc(vec[i]);
    	}
    	return result;
    }
    
    /** Unary operation taking two doubles
     * @author Walter Ho
     */
    private interface BinaryOp{
    	public double calc(double a, double b);
    }
    
    /** Performs a binary operation on corresponding elements in the two vectors 
     * and returns the result in a new vector
     * @param vec1 vector of values to be used as left-side operands
     * @param vec2 vector of values to be used as right-side operands
     * @param op binary operator to apply
     * @return new vector with results of binary operation
     */
    public static double[] binaryCalc(double vec1[], double vec2[], BinaryOp op){
    	double[] result = new double[vec1.length];
    	for(int i = 0; i< vec1.length; i++){
    		result[i] = op.calc(vec1[i], vec2[i]); 
    	}
    	return result;
    }
    
    /** Performs a binary operation on corresponding elements in the two 2D matrices 
     * and returns the result in a new matrix
     * @param mat1 matrix of values to be used as left-side operand
     * @param mat1 matrix of values to be used as right-side operand
     * @param op binary operator to apply
     * @return new matrix with results of binary operation
     */
    public static double[][] binaryCalc(double mat1[][], double mat2[][], BinaryOp op){
    	int rows = mat1.length;
    	double[][] result = new double[rows][];
    	for(int r = 0; r<rows; r++){
    		result[r]= binaryCalc(mat1[r], mat2[r], op);
    	}
    	return result;
    }
    
    /** Performs a binary operation on corresponding elements in the two 3D matrices 
     * and returns the result in a new matrix
     * @param mat1 matrix of values to be used as left-side operand
     * @param mat1 matrix of values to be used as right-side operand
     * @param op binary operator to apply
     * @return new matrix with results of binary operation
     */
    public static double[][][] binaryCalc(double mat1[][][], double mat2[][][], BinaryOp op){
    	int layers = mat1.length;
    	double[][][] result = new double[layers][][];
    	for(int l = 0; l<layers; l++){
    		result[l]= binaryCalc(mat1[l], mat2[l], op);
    	}
    	return result;
    }
    
    /** Performs vector addition and returns the result in a new vector
     * @param vec1 first vector to add
     * @param vec2 second vector to add
     * @return new vector with results of vector addition
     */
    public static double[] add(double[] vec1, double[] vec2){
    	if(vec1.length != vec2.length){
            System.out.println("Matrix.plus() array sizes dont match");
            return null;
        }
        BinaryOp add = (a, b)-> a+b;
        return binaryCalc(vec1, vec2, add);
    }
    
    /** Performs matrix addition and returns the result in a new matrix
     * @param mat1 first matrix to add
     * @param mat2 second matrix to add
     * @return new matrix with results of matrix addition
     */
    public static double[][] add(double[][] mat1, double[][] mat2){
    	if(mat1.length!=mat2.length||mat1.length!=mat2.length){
    		System.out.println("Matrix.plus() cannot take two matrixes of different sizes");
    	}
    	BinaryOp add = (a, b)->a+b;
    	return binaryCalc(mat1, mat2, add);
    }
    
    /** Performs vector subtraction and returns the result in a new vector
     * @param vec1 vector to subtract from
     * @param vec2 vector to subtract
     * @return new vector with results of vector subtraction
     */
    public static double[] subt(double[] vec1, double[] vec2){
    	if(vec1.length != vec2.length){
            System.out.println("Matrix.minus() array sizes dont match");
            return null;
        }
    	BinaryOp sub = (a, b)-> a-b;
        return binaryCalc(vec1, vec2, sub);
    }
    
    /** Performs matrix subtraction and returns the result in a new matrix
     * @param mat1 matrix to subtract from
     * @param mat2 matrix to subtract
     * @return new matrix with results of matrix subtraction
     */
    public static double[][] subt(double[][] mat1, double[][] mat2){
    	BinaryOp op = (a, b)->a-b;
    	return binaryCalc(mat1, mat2, op);
    }
    
    /** Performs 3D matrix subtraction and returns the result in a new matrix
     * @param mat1 matrix to subtract from
     * @param mat2 matrix to subtract
     * @return new matrix with results of matrix subtraction
     */
    public static double[][][] subt(double[][][] mat1, double[][][] mat2){
    	BinaryOp op = (a, b)->a-b;
    	return binaryCalc(mat1, mat2, op);
    }
    
    /** Performs element-wise multiplication (hadamark) of two vectors and returns 
     * the result in a new vector
     * @param vec1 first vector to use
     * @param vec2 second vector to use
     * @return new vector with results of hadamark
     */
    public static double[] hadamark(double[] vec1, double[] vec2){
    	if(vec1.length != vec2.length){
            System.out.println("Matrix.hadamark() array sizes dont match");
            return null;
        }
    	BinaryOp mult = (a, b)-> a*b;
        return binaryCalc(vec1, vec2, mult);
    }
    
    /** Performs element-wise multiplication (hadamark) of two matrixes and returns 
     * the result in a new vector
     * @param mat1 first matrixes to use
     * @param mat2 second matrixes to use
     * @return new matrixes with results of hadamark
     */
    public static double[][] hadamark(double[][] mat1, double[][] mat2){
    	if(mat1.length!=mat2.length||mat1.length!=mat2.length){
    		System.out.println("Matrix.hadamark() matrix sizes dont match");
    	}
    	int rows = mat1.length;
    	int cols = mat1[0].length;
    	double[][] result = new double[rows][cols];
    	for(int r = 0; r<rows; r++){
    		result[r]= hadamark(mat1[r], mat2[r]);
    	}
    	return result;
    }
    
    /** Transposes a 2D matrix and returns the result in a new matrix
     * @param mat matrix to transpose
     * @return new transposed matrix
     */
    public static double[][] transpose(double[][] mat){
    	int oldRows = mat.length;
    	int oldCols = mat[0].length;
    	double [][] result = new double[oldCols][oldRows];
    	for(int r = 0; r<oldRows; r++){
    		for(int c = 0; c<oldCols; c++){
    			result[c][r] = mat[r][c];
    		}
    	}
    	return result;
    }
    
    /** Scales a vector and returns the result in a new array
     * @param vector vector to be scaled
     * @param scale amount to scale by 
     * @return result of scale in a new array
     */
    public static double[] scale(double[] vector, double scale){
    	if(!Double.isFinite(scale)){
    		System.out.println("Matrix.scale() must take a finite scale");
    	}
    	UnaryOp op = (a)-> a*scale;
    	return unaryCalc(vector, op);
    }
    /** Scales a matrix and returns the result in a new matrix
     * @param mat matrix to be scaled
     * @param scale amount to scale by 
     * @return result of scale in a new matrix
     */
    public static double[][] scale(double[][] mat, double scale){
    	int rows = mat.length;
    	int cols = mat[0].length;
    	double[][] result = new double[rows][cols];
    	for(int r = 0; r<rows; r++){
    		result[r] = scale(mat[r], scale);
    	}
    	return result;
    }
    
    /** Sets the vector's values to zero
     * @param vec vector to clear
     */
    public static void clear(double[] vec){
    	for(int i =0; i< vec.length; i++) vec[i] =0;
    }
    /** Sets a matrix's values to zero
     * @param mat matrix to clear
     */
    public static void clear(double[][] mat){
    	for(double[] vec: mat) clear(vec);
    }
    /** Sets a 3D matrix's values to zero
     * @param mat matrix to clear
     */
    public static void clear(double[][][] rect){
    	for(double[][] mat: rect) clear(mat);
    }
    
    /** Fills an array with random Gaussian distributed double
     * @param vec vector to be filled
     */
    public static void gaussian(double[] vec){
    	Random rand = new Random(0);
    	for(int i =0; i< vec.length; i++) vec[i] = rand.nextGaussian();
    }
    /** Fills an matrix with random Gaussian distributed double
     * @param mat matrix to be filled
     */
    public static void gaussian(double[][] mat){
    	for(double[] vec: mat) gaussian(vec);
    }
    /** Fills a 3D matrix with random Gaussian distributed double
     * @param mat matrix to be filled
     */
    public static void gaussian(double[][][] rect){
    	for(double[][] mat: rect) gaussian(mat);
    }
    
    
}
