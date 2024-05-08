import numpy as np


class QArmUtilities():
    """QArm utilities, such as Forward/Inverse Kinematics, Differential Kinematics etc."""
    #region: QArm Parameters

    L_1 = 0.1400
    L_2 = 0.3500
    L_3 = 0.0500
    L_4 = 0.2500
    L_5 = 0.1500
    BETA = np.arctan(L_3/L_2)

    LAMBDA_1 = L_1
    LAMBDA_2 = np.sqrt(L_2**2 + L_3**2)
    LAMBDA_3 = L_4 + L_5
    #endregion

    @staticmethod
    def take_user_input_joint_space():
        """Use this method to take a user input for joint commands. Note that this method pauses execution until it returns. This is elegantly useful with the Position Mode on the QArm, which uses low level trajectory generation."""
        stringCmd = input("Where should the arm go? Enter floating point values separated by commas for the base, shoulder, elbow, wrist, and gripper.\nAll joints are floats, gripper is boolean. Type Ctrl+C to exit)\n")
        try:
            stringCmd = stringCmd.split(',')
            result = np.array([float(stringCmd[i]) for i in range(len(stringCmd))])
        except:
            print("Invalid entry. Going HOME instead. Please try again.")
            result = np.array([0, 0, 0, 0, 0])
        finally:
            return result

    @staticmethod
    def take_user_input_task_space():
        """Use this method to take a user input for end-effector position commands. Note that this method pauses execution until it returns. Note that this uses joint level trajectory generation, so the end-effector path will not be linear."""
        stringCmd = input("Where should the end-effector go? Enter floating point values separated by commas for the X, Y, Z, and gripper.\nAll values are floats, gripper is boolean. Type Ctrl+C to exit)\n")
        try:
            stringCmd = stringCmd.split(',')
            result = np.array([float(stringCmd[i]) for i in range(len(stringCmd))])
            if np.linalg.norm(result[0:3], ord=2) <= 0.25:
                print("Too close to base, going HOME instead. Please try again.")
                result = np.array([0.45, 0.00, 0.49, 0])
        except:
            print("Invalid entry. Going HOME instead. Please try again.")
            result = np.array([0.45, 0.00, 0.49, 0])
        finally:
            return result

    def qarm_forward_kinematics(self, phi):
        """ QUANSER_ARM_FPK v 1.0 - 30th August 2020

        REFERENCE:
        Chapter 3. Forward Kinematics
        Robot Dynamics and Control, Spong, Vidyasagar, 1989

        INPUTS:
        phi     : Alternate joint angles vector 4 x 1

        OUTPUTS:
        p4      : End-effector frame {4} position vector expressed in base frame {0}
        R04     : rotation matrix from end-effector frame {4} to base frame {0}"""

        # From phi space to theta space
        theta = phi
        theta[0] = phi[0]
        theta[1] = phi[1] + self.BETA - np.pi/2
        theta[2] = phi[2] - self.BETA
        theta[3] = phi[3]

        # Transformation matrices for all frames:

        # T{i-1}{i} = quanser_arm_DH(  a, alpha,  d,     theta )
        T01 = self.quanser_arm_DH(            0, -np.pi/2, self.LAMBDA_1,  theta[0] )
        T12 = self.quanser_arm_DH( self.LAMBDA_2,        0,            0,  theta[1] )
        T23 = self.quanser_arm_DH(            0, -np.pi/2,            0,  theta[2] )
        T34 = self.quanser_arm_DH(            0,        0, self.LAMBDA_3,  theta[3] )

        T02 = T01@T12
        T03 = T02@T23
        T04 = T03@T34

        # Position of end-effector Transformation

        # Extract the Position vector
        # p1   = T01(1:3,4);
        # p2   = T02(1:3,4);
        # p3   = T03(1:3,4);
        p4   = T04[0:3,3]

        # Extract the Rotation matrix
        # R01 = T01(1:3,1:3);
        # R02 = T02(1:3,1:3);
        # R03 = T03(1:3,1:3);
        R04 = T04[0:3,0:3]

        return p4, R04

    def qarm_inverse_kinematics(self, p, gamma, phi_prev):
        """
        QUANSER_ARM_IPK v 1.0 - 31st August 2020

        REFERENCE:
            Chapter 4. Inverse Kinematics
            Robot Dynamics and Control, Spong, Vidyasagar, 1989

        INPUTS:
            p: end-effector position vector expressed in base frame {0}
            gamma: wrist rotation angle gamma

        OUTPUTS:
            phiOptimal : Best solution depending on phi_prev
            phi: All four Inverse Kinematics solutions as a 4x4 matrix. Each col is a solution.
        """

        # Initialization
        theta   = np.zeros((4, 4), dtype=np.float64)
        phi = np.zeros((4, 4), dtype=np.float64)

        # Equations:
        # LAMBDA_2 cos(theta2) + (-LAMBDA_3) sin(theta2 + theta3) = sqrt(x^2 + y^2)
        #   A     cos( 2    ) +     C      sin(   2   +    3  ) =    D

        # LAMBDA_2 sin(theta2) - (-LAMBDA_3) cos(theta2 + theta3) = LAMBDA_1 - z
        #   A     sin( 2    ) -     C      cos(   2   +    3  ) =    H

        # Solution:
        def inv_kin_setup(p):
            A = self.LAMBDA_2
            C = -self.LAMBDA_3
            H =  self.LAMBDA_1 - p[2]
            D1 = -np.sqrt(p[0]**2 + p[1]**2)
            D2 =  np.sqrt(p[0]**2 + p[1]**2)
            F = (D1**2 + H**2 - A**2 - C**2)/(2*A)
            return A, C, H, D1, D2, F

        def solve_case_C_j2(j3, A, C, D, H):
            M = A + C*np.sin(j3)
            N = -C*np.cos(j3)
            cos_term = (D*M + H*N)/(M**2 + N**2)
            sin_term = (H - N*cos_term)/(M)
            j2 = np.arctan2(sin_term, cos_term)
            return j2

        A, C, H, D1, D2, F = inv_kin_setup(p)

        # Joint 3 solution:
        theta[2,0] = 2*np.arctan2( C + np.sqrt(C**2 - F**2) , F )
        theta[2,1] = 2*np.arctan2( C - np.sqrt(C**2 - F**2) , F )
        theta[2,2] = 2*np.arctan2( C + np.sqrt(C**2 - F**2) , F )
        theta[2,3] = 2*np.arctan2( C - np.sqrt(C**2 - F**2) , F )

        # Joint 2 solution:
        theta[1,0] = solve_case_C_j2(theta[2,0], A, C, D1, H)
        theta[1,1] = solve_case_C_j2(theta[2,1], A, C, D1, H)
        theta[1,2] = solve_case_C_j2(theta[2,2], A, C, D2, H)
        theta[1,3] = solve_case_C_j2(theta[2,3], A, C, D2, H)

        # Joint 1 solution:
        theta[0,0] = np.arctan2( p[1]/( self.LAMBDA_2 * np.cos( theta[1,0] ) - self.LAMBDA_3 * np.sin( theta[1,0] + theta[2,0] ) ) ,
                                 p[0]/( self.LAMBDA_2 * np.cos( theta[1,0] ) - self.LAMBDA_3 * np.sin( theta[1,0] + theta[2,0] )  ) )
        theta[0,1] = np.arctan2( p[1]/( self.LAMBDA_2 * np.cos( theta[1,1] ) - self.LAMBDA_3 * np.sin( theta[1,1] + theta[2,1] ) ) ,
                                 p[0]/( self.LAMBDA_2 * np.cos( theta[1,1] ) - self.LAMBDA_3 * np.sin( theta[1,1] + theta[2,1] )  ) )
        theta[0,2] = np.arctan2( p[1]/( self.LAMBDA_2 * np.cos( theta[1,2] ) - self.LAMBDA_3 * np.sin( theta[1,2] + theta[2,2] ) ) ,
                                 p[0]/( self.LAMBDA_2 * np.cos( theta[1,2] ) - self.LAMBDA_3 * np.sin( theta[1,2] + theta[2,2] )  ) )
        theta[0,3] = np.arctan2( p[1]/( self.LAMBDA_2 * np.cos( theta[1,3] ) - self.LAMBDA_3 * np.sin( theta[1,3] + theta[2,3] ) ) ,
                                 p[0]/( self.LAMBDA_2 * np.cos( theta[1,3] ) - self.LAMBDA_3 * np.sin( theta[1,3] + theta[2,3] )  ) )

        # Remap theta back to phi
        phi[0,:] = theta[0,:]
        phi[1,:] = theta[1,:] - self.BETA + np.pi/2
        phi[2,:] = theta[2,:] + self.BETA
        phi[3,:] = gamma*np.ones((4))

        phi = np.mod(phi + np.pi, 2*np.pi) - np.pi

        phiOptimal = phi[:,0]
        if np.linalg.norm(phiOptimal - phi_prev) > np.linalg.norm(phi[:,1] - phi_prev):
            phiOptimal = phi[:,1]
        if np.linalg.norm(phiOptimal - phi_prev) > np.linalg.norm(phi[:,2] - phi_prev):
            phiOptimal = phi[:,2]
        if np.linalg.norm(phiOptimal - phi_prev) > np.linalg.norm(phi[:,3] - phi_prev):
            phiOptimal = phi[:,3]

        return phi, phiOptimal

    def quanser_arm_DH(self, a, alpha, d, theta):

        """ QUANSER_ARM_DH
        v 1.0 - 26th March 2019

        REFERENCE:
        Chapter 3. Forward and Inverse Kinematics
        Robot Modeling and Control
        Spong, Hutchinson, Vidyasagar
        2006

        INPUTS:
        a       :   translation  : along : x_{i}   : from : z_{i-1} : to : z_{i}
        alpha   :      rotation  : about : x_{i}   : from : z_{i-1} : to : z_{i}
        d       :   translation  : along : z_{i-1} : from : x_{i-1} : to : x_{i}
        theta   :      rotation  : about : z_{i-1} : from : x_{i-1} : to : x_{i}
        (Standard DH Parameters are being used here)

        OUTPUTS:
        T       : transformation                   : from :     {i} : to : {i-1}"""

        # Rotation Transformation about z axis by theta
        T_R_z = np.array([[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

        # Translation Transformation along z axis by d
        T_T_z = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]], dtype=np.float64)

        # Translation Transformation along x axis by a
        T_T_x = np.array([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

        # Rotation Transformation about x axis by alpha
        T_R_x = np.array([[1, 0, 0, 0], [0, np.cos(alpha), -np.sin(alpha), 0], [0, np.sin(alpha), np.cos(alpha), 0], [0, 0, 0, 1]], dtype=np.float64)

        # For a transformation FROM frame {i} TO frame {i-1}: A
        T = T_R_z@T_T_z@T_T_x@T_R_x

        return T