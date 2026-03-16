import os
import numpy as np
import joblib
from django.shortcuts import render
from django.conf import settings

from .models import PropertyEvaluation

# \u2500\u2500 Load the trained model once at module level \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'rf_model.pkl')
model = joblib.load(MODEL_PATH)

# \u2500\u2500 Encoding maps (must match train_model.py) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
DISTRICT_MAP = {
    'Kentron': 1,
    'Arabkir': 2,
    'Avan': 3,
    'Shengavit': 4,
    'Erebuni': 5,
}

CONDITION_MAP = {
    'Old': 0,
    'Normal': 1,
    'New': 2,
}

# Armenian display labels
DISTRICT_DISPLAY = {
    'Kentron': 'Կենտրոն',
    'Arabkir': 'Արաբկիր',
    'Avan': 'Ավան',
    'Shengavit': 'Շենգավիտ',
    'Erebuni': 'Էրեբունի',
}

CONDITION_DISPLAY = {
    'New': 'Նոր վերանորոգված',
    'Normal': 'Լավ վիճակ',
    'Old': 'Վերանորոգման կարիք ունի',
}


def predict_price(request):
    """Handle GET \u2192 show form, POST \u2192 predict and show result."""

    if request.method == 'POST':
        # Extract form data
        area_sqm = float(request.POST.get('area_sqm'))
        rooms = int(request.POST.get('rooms'))
        floor = int(request.POST.get('floor'))
        district = request.POST.get('district')
        condition = request.POST.get('condition')

        # Encode categorical features
        district_encoded = DISTRICT_MAP[district]
        condition_encoded = CONDITION_MAP[condition]

        # Build feature array & predict
        features = np.array([[area_sqm, rooms, floor,
                              district_encoded, condition_encoded]])
        predicted_price = model.predict(features)[0]

        # Save to database
        evaluation = PropertyEvaluation.objects.create(
            area_sqm=area_sqm,
            rooms=rooms,
            floor=floor,
            district=district,
            condition=condition,
            predicted_price=round(predicted_price, 2),
        )

        return render(request, 'predictor/result.html', {
            'predicted_price': round(predicted_price, 2),
            'evaluation': evaluation,
            'district_display': DISTRICT_DISPLAY.get(district, district),
            'condition_display': CONDITION_DISPLAY.get(condition, condition),
        })

    # GET request
    return render(request, 'predictor/form.html')
