from django.db import models

# Create your models here.

def __str__(self): # admin에 제목으로 표시
        return self.title

class Image(models.Model):
    title = models.CharField(max_length=200,blank=True, null=True)
    pub_date = models.DateTimeField('date published',blank=True, null=True)

    #model.뭐뭐뭐Field(sdfs)
    def __str__(self): # admin에 제목으로 표시
        return self.title
    
class Pic(models.Model):
    fore_image = models.ForeignKey(Image, on_delete=models.CASCADE, null=True)
    image = models.ImageField(upload_to='images/',blank=True, null=True)
