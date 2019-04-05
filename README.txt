## Implementation of Feed-Forward Neural Network alongwith Backward Propagation
## SAKSHAM JAIN 2017MT10747

Training the Model:
Make a class object of Class My_model, passing the arguements [<array of numbers of neurons in hidden layers>] and Weight_Scale.

Choose Learning Rate at the top of the code (it has been made a global variable).

Then call method 'train' of the object, passing arguments X_train, Y_train, X_validation, Y_validation, Num_Epochs, Batch_Size,
Learning_Rate_Decay, Number_Train_Samples, Number_Validation_Samples.


For testing:
Call np.argmax(model.loss(data['X_test']), axis=1)
The Submission file will be generated and exported as 'submission.csv'

