from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import os

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

"""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import save_model
model = VGG16(weights='imagenet')
"""

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)
            
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)

            result = model.predict(img_array)
            print(decode_predictions(result))
            pred_result = decode_predictions(result)
            pred1 = (pred_result[0][0][1],pred_result[0][0][2]*100)
            pred2 = (pred_result[0][1][1],pred_result[0][1][2]*100)
            pred3 = (pred_result[0][2][1],pred_result[0][2][2]*100)
            pred4 = (pred_result[0][3][1],pred_result[0][3][2]*100)
            pred5 = (pred_result[0][4][1],pred_result[0][4][2]*100)
            prediction = 'dammy'
            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'prediction': prediction,'pred1_ctg': pred1[0], 'pred1_rt': pred1[1], 'pred2_ctg': pred2[0], 'pred2_rt': pred2[1], 'pred3_ctg': pred3[0], 'pred3_rt': pred3[1], 'pred4_ctg': pred4[0], 'pred4_rt': pred4[1], 'pred5_ctg': pred5[0], 'pred5_rt': pred5[1], 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})
