import pybullet as p
import numpy as np
import random
import cv2

useProgrammatic = 0
useTerrainFromPNG = 1
useDeepLocoCSV = 2

def create_field(heightfieldSource=0, meshScale=[0.01, 0.01, 0.01]):
  random.seed(10)
  textureId = -1
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

  if heightfieldSource==useProgrammatic:
    numHeightfieldRows = 256
    numHeightfieldColumns = 256
    heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
    for j in range (int(numHeightfieldColumns/2)):
      for i in range (int(numHeightfieldRows/2) ):
        height = random.uniform(0, 1)
        heightfieldData[2*i+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
        heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
    heightMap = np.resize(np.array(heightfieldData), [numHeightfieldRows, numHeightfieldColumns])
    terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=meshScale, heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
    terrain  = p.createMultiBody(0, terrainShape)
    p.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])
    p.changeVisualShape(terrain, -1, rgbaColor=[1,1,1,1])

  if heightfieldSource==useDeepLocoCSV:
    terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=meshScale,fileName = "my_envs/heightmaps/ground0.txt", heightfieldTextureScaling=128)
    terrain  = p.createMultiBody(0, terrainShape)
    p.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])
    p.changeVisualShape(terrain, -1, rgbaColor=[1,1,1,1])

  if heightfieldSource==useTerrainFromPNG:
    heightMap = cv2.imread("my_envs/heightmaps/wm_height_out.png")
    heightMap = cv2.cvtColor(heightMap, cv2.COLOR_BGR2GRAY)/255
    terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=meshScale,fileName = "my_envs/heightmaps/wm_height_out.png")
    textureId = p.loadTexture("my_envs/heightmaps/gimp_overlay_out.png")
    terrain  = p.createMultiBody(0, terrainShape)
    p.changeVisualShape(terrain, -1, textureUniqueId = textureId)
    
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
  
  return heightMap