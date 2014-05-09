#include "gpu_gng.h"
#include "cpu_gng.h"
#include "test.h"
#include "MNIST_GNG.h"


int main()
{
    //runTest();
    trainMNIST_GNG();
    yolo();


    cout << "DONE " << endl;
    cout << "Press enter to exit...";
    getchar();
    return 0;
}