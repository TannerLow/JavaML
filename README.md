# JavaML
Bare bones ML library for Java.

---

## How to Use in Other Projects

Add the following:  
* in your `build.gradle`
```
dependencies {
    implementation 'com.github.TannerLow:JavaML:0.2'
}
```
* in your `settings.gradle`
```
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}
```