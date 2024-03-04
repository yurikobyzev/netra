#=============================================================================================#
# Класс экземпляра Лимба
#=============================================================================================#
class tlimb:
    def __init__(self,center,center_pupil,radius,roi,limb_points):
        self.center = center
        self.center_pupil = center_pupil
        self.radius = radius
        self.roi = roi
        self.limb_points = limb_points
        pass
#=============================================================================================#
# Класс экземпляра Глазной щели
#=============================================================================================#
class teyelid:
    def __init__(self,center,radius,roi,pupil_center,eyelid_center,eyelid_points, limb_points):
        self.center = center
        self.radius = radius
        self.roi = roi
        self.pupil_center = pupil_center
        self.eyelid_center = eyelid_center
        self.eyelid_points = eyelid_points
        self.limb_points = limb_points
        pass
#=============================================================================================#