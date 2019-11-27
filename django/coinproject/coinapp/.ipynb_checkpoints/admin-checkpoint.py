from django.contrib import admin
from .models import Image, Pic

# Register your models here.
# Register your models here.

class PicInline(admin.TabularInline):
    model = Pic

class ImageAdmin(admin.ModelAdmin):
    inlines = [PicInline,]
# Register your models here.
admin.site.register(Image, ImageAdmin)
