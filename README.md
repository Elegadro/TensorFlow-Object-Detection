# Tensorflow Object Detection

1. `training-model.ipynb` dosyasını Google Colab üzerinde çalıştırın.\
    1.1. Tüm adımları istenilen şekilde yürütün.\
    1.2. Adımları yürütürken bazı dosyalar yüklemenizi veya kopyalamanız istenecektir. Bu dosyaları `Upload Files` klasöründe bulabilirsiniz. Dosyaları istenilen dizine yükleyebilirsiniz.\
    1.3. Özellikle Adım 4 aşamasını yaparken dikkatli olun.\
    1.4. Adım 9'da hata alırsanız Adım 9'u pas geçebilirsiniz.\
    1.5. Adım 10'u uyguladığınızda bilgisayarınıza `annotations.zip` ve `exported-models` adlı iki adet .zip dosyası indirilecektir.
    
2. İndirilen dosyaları zipten çıkartarak, Google Colab üzerinde oluşan dizinleri birebir taklit edin ve bu iki dosyayı `content/workspace/training_demo/` içine atın.

3. Eğitilen resimlerin gray-scale formatında olması durumunda (başka bir formatta da olabilir) rgb resim üzerinde görmek için `forgrayimg.py` dosyasını çalıştırın.<br />
3.1. Bu durumda iki adet dizin belirtmeniz gerekmektedir. Dosya içerisinde dizinleri belirtmeyi unutmayın!

4. Eğitilen resimler rgb (veya bgr) modunda olması durumunda `forbgrimg.py` dosyasını çalıştırın.\
    4.1. Dosya içerisinde resminizin dizinini belirtmeyi unutmayın!

5. Gerçek zamanlı obje tespiti (real-time) için `stream.py` veya `streamforonedim.py` dosyalarını çalıştırabilirsiniz.\
    5.1. `streamforonedim.py` dosyasını tek boyutlu eğitimler için (örneğin gray) kullanabilirsiniz.
