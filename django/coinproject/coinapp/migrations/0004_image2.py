# Generated by Django 2.2.7 on 2019-11-26 15:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('coinapp', '0003_pic'),
    ]

    operations = [
        migrations.CreateModel(
            name='Image2',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
            ],
        ),
    ]
