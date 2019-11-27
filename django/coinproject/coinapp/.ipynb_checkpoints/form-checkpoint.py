from django import forms
from .models import Image

#임의의 폼 형태는 Form 입력
class ImagePost(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['title']