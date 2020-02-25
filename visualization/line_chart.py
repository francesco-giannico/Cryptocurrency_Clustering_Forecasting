import matplotlib.pyplot as plt

"""def plot_train_and_validation_loss(history,output_folder,filename):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(output_folder+"model_train_validation_loss_"+str(filename.day)+".png",dpi=150)"""
from matplotlib.lines import Line2D
# plot train and validation loss across multiple runs
def plot_train_and_validation_loss(train,test,output_folder):
    plt.plot(train, color='blue', label='Train')
    plt.plot(test, color='orange', label='Validation')
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Train'),
                       Line2D([0], [0], color='orange', lw=2, label='Validation'),]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.savefig(output_folder+"model_train_validation_loss.png",dpi=150)