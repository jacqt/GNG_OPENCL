#include "gpu_gng.h"
#include "cpu_gng.h"
#include "test.h"
#include "MNIST_GNG.h"

int main()
{
    //runTest();
    trainMNIST_GNG();
    //yolo();



    cout << "DONE " << endl;
    int f;
    cout << "Enter any key and press ENTER to exit...";
    cin >> f;
    return 0;
}