#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
	cout << "argc: " << argc << endl;

	for(size_t i = 1; i < argc; i++){
		cout << "argv[" << i << "]: " << argv[i] << endl;
	}
}