
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten,Dense,UpSampling2D, concatenate
from loss_function import *

#size_image=(128,128,3)

def unet_model(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)
    
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    u6 = UpSampling2D((2, 2))(c4)
    u6 = concatenate([u6, c3])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c2])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c1])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c8)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

class UNet:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = unet_model(input_shape)

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def predict(self, image_batch):
        return self.model.predict(image_batch)
    
        
    def evaluate(self, image_batch, mask_batch):
        predictions = self.model.predict(image_batch)
        dice_score = dice(mask_batch, predictions)
        return dice_score.numpy()
    



class ShipModelClassifications:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        inputs = Input(self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def evaluate(self, image_batch, target_batch, threshold=0.5):
        predictions = self.model.predict(image_batch)
        binary_predictions = (predictions > threshold).astype(int)
        correct_predictions = np.sum(binary_predictions == target_batch)
        total_images = len(target_batch)
        accuracy = correct_predictions / total_images
        return accuracy


class ClassificationUNetDetection:
    def __init__(self, classification_model, segmentation_model, threshold=0.5):
        self.classification_model = classification_model
        self.segmentation_model = segmentation_model
        self.threshold = threshold

    def dice_for_1_image(self, targets, inputs, smooth=1e-6):
        targets = tf.cast(targets, tf.float32)
        inputs = tf.cast(inputs, tf.float32)
        if len(targets.shape) == 3:
            targets = tf.expand_dims(targets, axis=-1)
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=-1)

        axis = [1, 2, 3]
        intersection = K.sum(targets * inputs, axis=axis)
        dice_f = (2. * intersection + smooth) / (K.sum(targets, axis=axis) + K.sum(inputs, axis=axis) + smooth)
        return dice_f

    def predict_ship(self, image):
        input_image = np.expand_dims(image, axis=0)
        ship_prob = self.classification_model.predict(input_image)
        has_ship = ship_prob[0] >= self.threshold
        if has_ship:
            mask = self.segmentation_model.predict(input_image)
            return True, mask[0]
        else:
            return False, None

    def evaluate(self, dataset):
        dice_scores = []

        for img, true_mask in dataset:
            has_ship, predicted_mask = self.predict_ship(img)

            if has_ship:
                dice_score = self.dice_for_1_image(predicted_mask, true_mask).numpy()
                dice_scores.append(dice_score)

        mean_dice = np.mean(dice_scores)
        return mean_dice
