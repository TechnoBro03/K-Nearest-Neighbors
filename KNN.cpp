#include <iostream> /// Text output
#include <string> /// Strings
#include <sstream> /// File reading
#include <fstream> /// File reading
#include <math.h> /// For math
#include <float.h> /// For DBL_MAX

#define index(pointX,attY) (pointX*NUMATTRIB + attY) /// Used to index 1D array

/// Constants
const int NUMPOINTSTRAIN = 1469;
const int NUMPOINTSTEST = 3429;
const int NUMATTRIB = 12;

const int K = 5;

const std::string PATHTRAIN = "wine_train.txt";
const std::string PATHTEST = "wine_test.txt";

/// A point is used in my implementation of argmin
struct point
{
    int index;
    double distance;
    double label;
    point()
    {
        index = -1;
        distance = DBL_MAX;
        label = 0;
    }
    point(int i, double d, double l)
    {
        index = i;
        distance = d;
        label = l;
    }
};

void readFile(std::string path, double* arr, int numPoints)
{
	std::ifstream stream;
	stream.open(path);
    std::string line;

	for (int i = 0; i < numPoints; i++)
    {
		std::getline(stream, line);
		std::stringstream ss(line);
        std::string substr;

        for(int j = 0; j < NUMATTRIB; j++)
        {
            std::getline(ss, substr, ',');
            arr[index(i,j)] = std::stod(substr);
        }
	}
	stream.close();
	return;
}

double getDistance(double* a, double* b, int pointA, int pointB)
{
    double distance = 0;
    /// Finds absolute difference between all features and squares it
    for(int i = 0; i < NUMATTRIB-1; i++)
    {
        double c = std::abs(a[index(pointA, i)] - b[index(pointB, i)]);
        distance += std::pow(c, 2);
    }
    /// No need to actually calculate square root, has no effect.
    return distance;
}

double getMode(point* arr, int len)
{
    int maxCount = 0;
    double maxValue = 0;
    int smallestIndex = 0;

    /// Count how many occurrences of each label
    for (int i = 0; i < len; i++)
    {
        int count = 0;
        for (int j = 0; j < len; j++)
        {
            if (arr[i].label == arr[j].label)
            {
                count++;
            }
        }

        /// If a new max is found, change it
        if (count > maxCount)
        {
            maxCount = count;
            maxValue = arr[i].label;
            smallestIndex = i;
        }
        
        /// If a tie is found, find the smallest index among points
        if(count == maxCount)
        {
            for(int j = 0; j < len; j++)
            {
            if (arr[i].label == arr[j].label)
            {
                if(arr[smallestIndex].index > arr[j].index)
                {
                    smallestIndex = j;
                }
            }
            }
            maxValue = arr[smallestIndex].label;
        }
    }
    return maxValue;
}

void myKNN(double* trainData, double* testData, int k)
{
    std::cout << "K Value = " << k << std::endl;
    std::cout << "Index\tPredicted\tActual" << std::endl;

    int numCorrect = 0;
    /// Keep track of only the min K points, for argmin
    point* minKPoints = new point[k];

    /// For each point in testData
    for(int pointI = 0; pointI < NUMPOINTSTEST; pointI++)
    {
        /// Init minKPoint[] to max distance
        int largestIndex = 0;
        for(int i = 0; i < k; i++)
        {
            minKPoints[i].distance = DBL_MAX;
        }
        
        /// For each point in trainData
        for(int pointJ = 0; pointJ < NUMPOINTSTRAIN; pointJ++)
        {

            /// Below is my implementation of argmin
            /// Space complexity is O(k)
            /// Time complexity is worst case O(k), only when "inserting" a new "closest" point
            /// Time complexity is best case O(1), if point is not less than any other k-min points
            /// No large vector/array to hold all distances, no sorting the vector/array at the end

            double d = getDistance(testData, trainData, pointI, pointJ);

            /// Create a new point to keep track if its index, distance, and label
            point p(pointJ, d, trainData[index(pointJ, NUMATTRIB-1)]);

            /// If point p has a smaller distance than the largest distance in minKPoints[], change
            if(p.distance < minKPoints[largestIndex].distance)
            {
                minKPoints[largestIndex] = p;

                /// Find new largest index within minKPoints[]
                largestIndex = 0;
                for(int s = 0; s < k; s++)
                {
                    if(minKPoints[largestIndex].distance < minKPoints[s].distance)
                    {
                        largestIndex = s;
                    }
                }
            }
        }

        /// Get predicted label and compare to actual
        double predicted = getMode(minKPoints, k);
        double actual = testData[index(pointI,NUMATTRIB-1)];
        if(predicted == actual)
        {
            numCorrect++;
        }
        std::cout << pointI+1 << "\t" << predicted << "\t\t" << actual << std::endl;
    }

    delete[] minKPoints;

    std::cout << "\nNumber of correctly classified instances: " << numCorrect << "\nTotal number of instances in the test set: "
    << NUMPOINTSTEST << "\nPercentage of correctly classified instances: " << std::round((double)numCorrect/NUMPOINTSTEST*10000)/100 << "%" << std::endl;
}

int main()
{
    /// Store data in 1D array (more space and time efficient)
    double* trainData = new double[NUMPOINTSTRAIN*NUMATTRIB];
    double* testData = new double[NUMPOINTSTEST*NUMATTRIB];

    try
    {
    readFile(PATHTRAIN, trainData, NUMPOINTSTRAIN);
    readFile(PATHTEST, testData, NUMPOINTSTEST);

    myKNN(trainData, testData, K);
    }
    catch(...)
    {
        std::cout << "An error occured" << std::endl;
    }

    delete[] trainData;
    delete[] testData;

    /// Keep terminal from closing
    std::cin.get();

    return 0;
}
