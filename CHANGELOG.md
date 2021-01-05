# Change Log

## 2021-01-05
- Fixed problematic annotations for three chairs. Removed reflection planes of legs in swivel chairs.
## 2020-12-12
- Done code cleanup. Now it's much easier to run and read benchmark code.

## 2020-06-15
- **Full dataset is available now!** Note: some of knives are misaligned (where x-axis is flipped) in the orginal ShapeNet. We picked those misaligned ones and stored them into **knife_misaligned.txt**.

## 2020-04-18
- We've split the semantic ids of normal chairs (**four** legs) and swivel chairs (**five** legs with rotational symmetries). **Before**, the four legs of normal chairs have semantic id: **10, 11, 12, 13** and the five legs of swivel chairs have semantic id: **10, 11, 12, 13, 14**. **Now**, the four legs of normal chairs have semantic id: **17, 18, 19, 20** and semantic ids of swivel chairs are the same as before.

## 2020-04-17
- cleaned saliency experiments code
- cleaned correspondence experiments code

## 2020-04-03

- Add colored pcd and ply file on ShapeNet.v2
- Replace json annotations with updated information

## 2020-04-01

- First release on ShapeNet.v2 airplane, chair and table classes
