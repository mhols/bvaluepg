import numpy as np


class Coordinates:
    def __init__(self, center_lon, center_lat, rotation_angle=0):
        self.center_lon = center_lon
        self.center_lat = center_lat
        self.rotation_angle = rotation_angle

    def lonlat_to_xy(self, lon, lat):
        """
        Convert longitude and latitude to x and y coordinates in kilometers.
        """
        # Constants
        R = 6371.0  # Radius of the Earth in kilometers

        # Convert degrees to radians
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        center_lon_rad = np.radians(self.center_lon)
        center_lat_rad = np.radians(self.center_lat)

        # Calculate x and y coordinates
        x = R * (lon_rad - center_lon_rad) * np.cos(center_lat_rad)
        y = R * (lat_rad - center_lat_rad)

        return x, y

    def xy_to_lonlat(self, x, y):
        """
        Convert x and y coordinates in kilometers back to longitude and latitude.
        """
        # Constants
        R = 6371.0  # Radius of the Earth in kilometers

        # Convert radians to degrees
        center_lon_rad = np.radians(self.center_lon)
        center_lat_rad = np.radians(self.center_lat)

        # Calculate longitude and latitude
        lon = np.degrees(x / (R * np.cos(center_lat_rad))) + self.center_lon
        lat = np.degrees(y / R) + self.center_lat

        return lon, lat

    def rotate_coordinates(self, x, y):
        """
        Rotate coordinates by the specified rotation angle.
        """
        angle_rad = np.radians(self.rotation_angle)
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

        return x_rot, y_rot

    def inverse_rotate_coordinates(self, x_rot, y_rot):
        """
        Inverse rotate coordinates by the specified rotation angle.
        """
        angle_rad = np.radians(-self.rotation_angle)
        x = x_rot * np.cos(angle_rad) - y_rot * np.sin(angle_rad)
        y = x_rot * np.sin(angle_rad) + y_rot * np.cos(angle_rad)

        return x, y

    def lonlat_to_rotated_xy(self, lon, lat):
        """
        Convert longitude and latitude to rotated x and y coordinates in kilometers.
        """
        x, y = self.lonlat_to_xy(lon, lat)
        return self.rotate_coordinates(x, y)

    def rotated_xy_to_lonlat(self, x_rot, y_rot):
        """
        Convert rotated x and y coordinates in kilometers back to longitude and latitude.
        """
        x, y = self.inverse_rotate_coordinates(x_rot, y_rot)
        return self.xy_to_lonlat(x, y)

    def get_binned_data_in_rotated_coordinates(self, lon, lat, dkm):
        """
        Get binned data in rotated coordinates. 
        """
        x_rot, y_rot = self.lonlat_to_rotated_xy(lon, lat)
        nbinx = int(np.ceil((x_rot.max() - x_rot.min()) / dkm))
        nbiny = int(np.ceil((y_rot.max() - y_rot.min()) / dkm))

        i = np.digitize(x_rot, x_rot.min() + (np.arange(nbinx + 1)) * dkm) - 1
        j = np.digitize(y_rot, y_rot.min() + (np.arange(nbiny + 1)) * dkm) - 1

        ## count the number of events in each bin
        counts = np.zeros((nbinx, nbiny), dtype=int)
        for ii in range(nbinx):
            for jj in range(nbiny):
                counts[ii, jj] = np.sum((i == ii) & (j == jj))

        return i, j, nbinx, nbiny, x_rot, y_rot, counts, (x_rot.min(), x_rot.min() + nbinx * dkm, y_rot.min(), y_rot.min() + nbiny * dkm)
    

    

Italy_Coordinates = Coordinates(center_lon=12.5, center_lat=42.5, rotation_angle=-45)



    