package weka.classifiers.trees;

import weka.core.*;
import java.util.Arrays;
import java.util.Collections;

import javax.swing.*;
import java.awt.*;

/**
 * A splitter for using the best infogain-based bivariate split on numeric attributes.
 */
public class BivariateNumericAttributeSplitter extends Splitter {
    
    /** The natural logarithm of 2 */
    public static final double log2 = Math.log(2);

    /** The first attribute that will be used for splitting */
    private Attribute attX1;

    /** The second attribute that will be used for splitting */
    private Attribute attX2;
    
    /** The first coordinate for the split */
    private double x1;
    
    /** The second coordinate for the split */
    private double x2;
    
    /** The radius for the split */
    private double r;

    /** The split quality: the information gain */
    private double splitQuality;
    
    /**
     * Finding the best bivariate split across attributes using information gain.
     */
    public void findSplit(Instances data) {

	// Class distribution of all instances that reach this node.
	int[] initialCounts = new int[data.numClasses()];
	for (Instance instance : data) {
	    initialCounts[(int) instance.classValue()]++;
	}

	// A circle is symmetric in its two axes, so only unordered pairs of
	// (non-class) attributes need to be examined.
	java.util.List<Attribute> attributes = Collections.list(data.enumerateAttributes());
	for (int i = 0; i < attributes.size(); i++) {
	    for (int j = i + 1; j < attributes.size(); j++) {
		findBestSplit(data, attributes.get(i), attributes.get(j), initialCounts);
	    }
	}
    }

    /**
     * Decides whether the given instance belongs to the first branch or not.
     * An instance is in the first (left) branch when it lies inside the circle,
     * i.e. its distance to the centre is smaller than the radius.
     */
    public boolean firstSubset(Instance instance) {
	double d1 = instance.value(attX1) - x1;
	double d2 = instance.value(attX2) - x2;
	return d1 * d1 + d2 * d2 < r * r;
    }

    /**
     * Returns the information gain of the best split found.
     */
    public double splitQuality() {
	return splitQuality;
    }

    /**
     * Returns a string representation of the condition for a branch.
     */
    public String toString(boolean leftBranch, int decimalPlaces) {
        return "Attributes: " + attX1.name() + "," + attX2.name() +
	    (leftBranch ? " inside " : " outside ") + "circle with center (" +
	    String.format("%." + decimalPlaces + "f", x1) + "," +
	    String.format("%." + decimalPlaces + "f", x2) + ") and radius " +
	    String.format("%." + decimalPlaces + "f", r); 
    }
    
    /**
     * Plot the split, along with a scatterplot of the relevant data.
     */
    public void plot(Instances data, int decimalPlaces) {
	Color[] colours = { Color.blue, Color.red, Color.green,
	    Color.cyan, Color.pink, new Color(255, 0, 255), Color.orange,
	    new Color(255, 0, 0), new Color(0, 255, 0), Color.white };
	int padding = 50;
	double minX = data.stream().mapToDouble(p -> p.value(attX1)).min().getAsDouble();
	double maxX = data.stream().mapToDouble(p -> p.value(attX1)).max().getAsDouble();
	double minY = data.stream().mapToDouble(p -> p.value(attX2)).min().getAsDouble();
	double maxY = data.stream().mapToDouble(p -> p.value(attX2)).max().getAsDouble();
	JFrame jf = new JFrame("X1: " + attX1.name() + " Range: " +
			       String.format("%." + decimalPlaces + "f", minX) + "," +
			       String.format("%." + decimalPlaces + "f", maxX) +
			       " X2: " + attX2.name() + " Range: " +
			       String.format("%." + decimalPlaces + "f", minY) + "," +
			       String.format("%." + decimalPlaces + "f", maxY));
	jf.add(new JPanel() {
		private int getX(double v) {
		    return (int) ((v - minX) / (maxX - minX) * (getWidth() - 2 * padding)) + padding;
		}
		private int getY(double v) {
		    return (int) ((maxY - v) / (maxY - minY) * (getHeight() - 2 * padding)) + padding;
		}
		private int getXradius(double v) {
		    return (int) (v / (maxX - minX) * (getWidth() - 2 * padding));
		}
		private int getYradius(double v) {
		    return (int) (v / (maxY - minY) * (getHeight() - 2 * padding));
		}
		protected void paintComponent(Graphics g) {
		    super.paintComponent(g);
		    for (Instance i : data) {
			g.setColor(colours[((int)i.classValue()) % colours.length]);
			g.fillOval(getX(i.value(attX1)) - 4, getY(i.value(attX2)) - 4, 8, 8);
		    }
		    g.drawOval(getX(x1) - getXradius(r), getY(x2) - getYradius(r), 2 * getXradius(r), 2 * getYradius(r));
		}
	    });
	jf.setSize(600, 600);
	jf.setVisible(true);
	JOptionPane pane = new JOptionPane(null);
	JDialog d = pane.createDialog(jf, null);
	d.setLocationRelativeTo(jf);
	d.setLocation(0,0);
	d.setSize(new Dimension(10,30));
	d.setVisible(true);
	jf.dispose();
    }
    
    /**
     * Finds the best circular split based on the given reduced data.
     */
    private void findBestSplit(Instances data, Attribute a1, Attribute a2, int[] initialCounts) {
	int n = data.numInstances();
	int numClasses = data.numClasses();

	// Squared distances (reused for each centre) and an index array to sort.
	final double[] dist = new double[n];
	Integer[] order = new Integer[n];

	// Every instance in the (a1, a2) plane is a candidate circle centre.
	for (int c = 0; c < n; c++) {
	    double cx1 = data.instance(c).value(a1);
	    double cx2 = data.instance(c).value(a2);
	    for (int i = 0; i < n; i++) {
		double dx = data.instance(i).value(a1) - cx1;
		double dy = data.instance(i).value(a2) - cx2;
		dist[i] = dx * dx + dy * dy;
		order[i] = i;
	    }
	    Arrays.sort(order, (p, q) -> Double.compare(dist[p], dist[q]));

	    // Grow the radius: instances move from outside (right) to inside (left)
	    // one distinct-distance group at a time. Score each boundary.
	    int[] countsInside = new int[numClasses];
	    int[] countsOutside = Arrays.copyOf(initialCounts, numClasses);
	    double prevDist = -1.0;
	    for (int k = 0; k < n; k++) {
		int i = order[k];
		double d = dist[i];
		if (d > prevDist) {
		    if (prevDist >= 0.0) {
			double quality = informationGain(countsInside, countsOutside);
			if (quality > splitQuality) {
			    splitQuality = quality;
			    attX1 = a1;
			    attX2 = a2;
			    x1 = cx1;
			    x2 = cx2;
			    // Representative radius: midpoint of the two consecutive
			    // actual (non-squared) distances rmin and rmax.
			    r = (Math.sqrt(prevDist) + Math.sqrt(d)) / 2.0;
			}
		    }
		    prevDist = d;
		}
		int label = (int) data.instance(i).classValue();
		countsInside[label]++;
		countsOutside[label]--;
	    }
	}
    }
    
    /**
     * Helper method for computing entropy.
     */
    public static double lnFunc(int num){
        return (num <= 0) ? 0 : num * Math.log(num);
    }
    
    /**
     * Computes the base-2 information gain for the given array of counts.
     */
    public static double informationGain(int[] leftCounts, int[] rightCounts) {
	double sumOfNLogNLeft = 0, sumOfNLogNRight = 0, sumOfNLogNTotal = 0;
	int sumLeft = 0, sumRight = 0, sumTotal = 0;
	for (int i = 0; i < leftCounts.length; i++) {
	    sumOfNLogNLeft -= lnFunc(leftCounts[i]);
	    sumLeft += leftCounts[i];
	    sumOfNLogNRight -= lnFunc(rightCounts[i]);
	    sumRight += rightCounts[i];
	    sumOfNLogNTotal -= lnFunc(leftCounts[i] + rightCounts[i]);
	}
	sumTotal = sumLeft + sumRight;
        return (sumTotal <= 0) ? 0 :
	    (sumOfNLogNTotal + lnFunc(sumTotal) -
	     (sumOfNLogNLeft + lnFunc(sumLeft) + sumOfNLogNRight + lnFunc(sumRight))) /
	    (sumTotal * log2);
    }
}
