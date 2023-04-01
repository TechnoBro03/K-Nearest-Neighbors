#include <iostream> /// Text output
#include <string> /// Strings
#include <sstream> /// File reading
#include <fstream> /// File reading
#include <math.h> /// For math
#include <queue> /// For priority queue (argmin)
#include <float.h> /// For DBL_MAX
#include <chrono> /// For timing

#define index(x,y) (x + y*DIM)

/// Constants
const int NUMTRAIN = 1469;
const int NUMTEST = 3425;
const int DIM = 12;

const std::string PATHTRAIN = "wine_train.txt";
const std::string PATHTEST = "wine_test.txt";

void readFile(std::string path, double* arr, int numRows)
{
    std::ifstream stream;
    stream.open(path);
    std::string line;

    for (int i = 0; i < numRows; i++)
    {
        std::getline(stream, line);
        std::stringstream ss(line);
        std::string substr;

        for(int j = 0; j < DIM; j++)
        {
            std::getline(ss, substr, ',');
            arr[index(j,i)] = std::stod(substr);
        }
    }
    stream.close();
    return;
}

double getDistance(double* a, double* b, int pointA, int pointB)
{
    double distance = 0;
	/// Finds absolute difference between all features and squares it
    for(int dim = 0; dim < DIM-1; dim++)
    {
        double c = std::abs(a[index(dim, pointA)] - b[index(dim, pointB)]);
        distance += std::pow(c, 2);
    }
    /// No need to actually calculate square root, has no effect.
    return distance;
}

/// Used for argmin
/// A point keeps track of distance, index, and label
struct point
{
    int index;
    double distance;
    double label;
    point(int index, double distance, double label)
    {
        this->index = index;
        this->distance = distance;
        this->label = label;
    }
};

/// Used for argmin priority queue comparisons
class Compare
{
    public:
    bool operator() (point a, point b)
    {
        if(a.distance < b.distance)
        {
            return true;
        }
        return false;
    }
};

double getMode(std::priority_queue<point, std::vector<point>, Compare>& pq, int k)
{
    /// Empty priority queue into vector for ease of use
    std::vector<point> arr;
    while (!pq.empty()) {
        arr.push_back(pq.top());
        pq.pop();
    }

    int maxCount = 0;
    double maxValue = 0;
    int smallestIndex = 0;

    for(int i = 0; i < k; i++)
    {
        int count = 0;
        int sIFL = i;
        for(int j = 0; j < k; j++)
        {
            if(arr[i].label == arr[j].label)
            {
                count++;
                /// Keep track of smallest index for this label (sIFL)
                if(arr[j].index < arr[sIFL].index)
                {
                    sIFL = j;
                }
            }
        }
        if(count > maxCount)
        {
            maxCount = count;
            maxValue = arr[i].label;
            smallestIndex = sIFL;
        }
        /// If there is a tie, pick the point with the smallest index
        if(count == maxCount)
        {
            if(arr[sIFL].index < arr[smallestIndex].index)
            {
                smallestIndex = sIFL;
            }
            maxValue = arr[smallestIndex].label;
        }
    }
    return maxValue;
}

void myKNN(double* trainData, double* testData, int k)
{
    std::cout << "\n\tK Value = " << k << std::endl;

    int numCorrect = 0;

    /// Implementation of argmin
    /// Space complexity: O(k)
    /// Time complexity: O(N*logK)
    ///     std::sort() time complexity: O(N*logN)
    /// Keep track of only the k min points
    std::priority_queue<point, std::vector<point>, Compare> kMin;

    /// For each point in testData
    for(int pointI = 0; pointI < NUMTEST; pointI++)
    {
        /// Init priority queue
        for(int i = 0; i < k; i++)
        {
            kMin.push(point(0, DBL_MAX, 0));
        }
        
        /// For each point in trainData
        for(int pointJ = 0; pointJ < NUMTRAIN; pointJ++)
        {
            double d = getDistance(testData, trainData, pointI, pointJ);

            /// Create a new point to keep track if its index, distance, and label
            point p(pointJ, d, trainData[index(DIM-1, pointJ)]);

            /// If point p has a smaller distance than the largest distance in kMin
            if(p.distance < kMin.top().distance)
            {
                kMin.pop(); /// Remove largest element: O(logN)
                kMin.push(p); /// Add new element: O(logN)
            }
        }

        /// Get predicted label and compare to actual
        double predicted = getMode(kMin, k);
        double actual = testData[index(DIM-1, pointI)];
        if(predicted == actual)
        {
            numCorrect++;
        }
    }

    std::cout << "\tNumber of correctly classified instances: " << numCorrect << "\n\tTotal number of instances in the test set: "
    << NUMTEST << "\n\tPercentage of correctly classified instances: " << (double)numCorrect/NUMTEST*100 << "%" << std::endl;
}

int main()
{
    /// Store data in 1D array (more space and time efficient)
    double* trainData = new double[NUMTRAIN*DIM];
    double* testData = new double[NUMTEST*DIM];

    try
    {
        /// Read files
        readFile(PATHTRAIN, trainData, NUMTRAIN);
        readFile(PATHTEST, testData, NUMTEST);

        /// Get user input for value of k
        while(true)
        {
            int k;
            std::cout << "Enter value for k (0 to exit): ";
            std::cin >> k;
            if(k == 0) break;

            /// Start clock
            std::chrono::time_point<std::chrono::system_clock> start, end;
            start = std::chrono::system_clock::now();

            myKNN(trainData, testData, k);

            /// Stop clock
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            
            std::cout << "\tElapsed time: " << elapsed_seconds.count() << "s\n" << std::endl;
        }
    }
    catch(...)
    {
        std::cout << "An error occured" << std::endl;
    }

    delete[] trainData;
    delete[] testData;

    return 0;
}
