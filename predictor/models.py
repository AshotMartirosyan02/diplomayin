from django.db import models


class PropertyEvaluation(models.Model):
    """Stores each prediction request and its result."""

    DISTRICT_CHOICES = [
        ('Kentron', 'Կենտրոն'),
        ('Arabkir', 'Արաբկիր'),
        ('Avan', 'Ավան'),
        ('Shengavit', 'Շենգավիտ'),
        ('Erebuni', 'Էրեբունի'),
    ]

    CONDITION_CHOICES = [
        ('New', 'Նոր վերանորոգված'),
        ('Normal', 'Լավ վիճակ'),
        ('Old', 'Վերանորոգման կարիք ունի'),
    ]

    area_sqm = models.FloatField(verbose_name='Մակերեսություն (m²)')
    rooms = models.IntegerField(verbose_name='Սենյակների քանակ')
    floor = models.IntegerField(verbose_name='Հարկ')
    district = models.CharField(max_length=50, choices=DISTRICT_CHOICES)
    condition = models.CharField(max_length=50, choices=CONDITION_CHOICES)
    predicted_price = models.FloatField(verbose_name='Կանխատեսվող Վարձավճար')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return (
            f"{self.get_district_display()} | {self.area_sqm} m\u00b2 | "
            f"{self.rooms} \u057d\u0565\u0576\u0575\u0561\u056f \u2192 {self.predicted_price:,.0f} AMD"
        )
