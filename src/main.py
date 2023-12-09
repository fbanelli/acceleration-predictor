from dataset_prepare import dataset_prepare
from networkRNN import trainRNN
from tester import validateandtest

ground_truth_poses_file_name = 'cloverData/groundTruthPoses.csv'
rotor_rpm_file_name = 'cloverData/blackbird_slash_rotor_rpm.csv'
output_directory = 'results/'
window_size = 10
epochs = 1000
batch_size = 32

def main():

    filtered_acc, rpms = dataset_prepare(ground_truth_poses_file_name, rotor_rpm_file_name)
    model = trainRNN(rpms, filtered_acc, window_size, epochs, batch_size, 'model/newmodel.h5')
    validateandtest(model, rpms, filtered_acc, output_directory)
    
if __name__ == "__main__":
    main()
