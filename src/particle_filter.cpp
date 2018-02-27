/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

        num_particles = 50;	
	
	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) 
        {
             double sample_x, sample_y, sample_theta;
             Particle par;
              
	     par.id = i;
             par.x = dist_x(gen);
	     par.y = dist_y(gen);
	     par.theta = dist_theta(gen);
             par.weight = 1;

             particles.push_back(par);
        }

        is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	// http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	// http://www.cplusplus.com/reference/random/default_random_engine/

        normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

        for (int i = 0; i< num_particles; i++)
        {
             if( fabs(yaw_rate) > 0.0001)
             {
                 particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
                 particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
                 particles[i].theta += yaw_rate * delta_t;
             }
             else
             {
                 particles[i].x += velocity * delta_t* cos(particles[i].theta);
                 particles[i].y += velocity * delta_t* sin(particles[i].theta);
             }
             
             // add noise
             particles[i].x += dist_x(gen);
             particles[i].y += dist_y(gen);
             particles[i].theta += dist_theta(gen);          
        }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) 
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
        
        for (unsigned int z=0; z< observations.size(); z++) 
        {
            int landmark_id = -1;
            double dist_min = numeric_limits<double>::max();

            for (unsigned int l=0; l< predicted.size(); l++)
            {
                 double distance = dist(observations[z].x, observations[z].y, predicted[l].x, predicted[l].y);
                 if(distance < dist_min)
                 {
                    dist_min = distance;
                    landmark_id = predicted[l].id;
                 }
            }

            observations[z].id = landmark_id;
        }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> &observations, const Map &map_landmarks) 
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
        
        double weight_Sum = 0;      
        vector<LandmarkObs> predicted;

        for (int i = 0; i< num_particles; i++)
        {    
             predicted.clear();

             //
             // predict landmark observation for each particle
             //
             for (unsigned int z=0; z< map_landmarks.landmark_list.size(); z++) 
             {
                  LandmarkObs temp;
                  temp.x = map_landmarks.landmark_list[z].x_f;
                  temp.y = map_landmarks.landmark_list[z].y_f;
                  temp.id = map_landmarks.landmark_list[z].id_i;

                  if(dist(temp.x, temp.y, particles[i].x, particles[i].y) <= sensor_range)
                  {
                     predicted.push_back(temp);
                  }
             }

             vector<LandmarkObs> transformed_observations;
             for (unsigned int z = 0; z < observations.size(); z++) 
             {
                  double t_x = cos(particles[i].theta)*observations[z].x - sin(particles[i].theta)*observations[z].y + particles[i].x;
                  double t_y = sin(particles[i].theta)*observations[z].x + cos(particles[i].theta)*observations[z].y + particles[i].y;
                  transformed_observations.push_back(LandmarkObs{ observations[z].id, t_x, t_y });
             }

             dataAssociation(predicted, transformed_observations);

             particles[i].weight = 1;

             //
             //run over current observation vector:
             //
             for (unsigned int z=0; z< transformed_observations.size(); z++) 
             {
                  double sig_x = std_landmark[0];
                  double sig_y = std_landmark[1];
                  double x_obs = transformed_observations[z].x;
                  double y_obs = transformed_observations[z].y;
                  double mu_x;
                  double mu_y;

                  for (unsigned int k = 0; k < predicted.size(); k++)
                  {
                      if (predicted[k].id == transformed_observations[z].id) 
                      {
                          mu_x = predicted[k].x;
                          mu_y = predicted[k].y;
                      }
                  }
                  
                  //
                  // calculate normalization term
                  //
                  double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));

                  //
                  // calculate exponent
                  //
                  double exponent= ((x_obs - mu_x)*(x_obs - mu_x))/(2 * sig_x*sig_x) + ((y_obs - mu_y)*(y_obs - mu_y))/(2 * sig_y*sig_y);

                  //
                  // calculate weight using normalization terms and exponent
                  //
                  particles[i].weight *= gauss_norm * exp(-exponent); 
              }
               
              weight_Sum += particles[i].weight;
         }

cout << "weightSum: " << weight_Sum << endl;

         //
         // normalize the weight between (0,1)
         //
         weights.clear();
         for (int i = 0; i< num_particles; i++)
         {
              weights.push_back(particles[i].weight/weight_Sum);
         }
}

void ParticleFilter::resample() 
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
        
        // generate random starting index for resampling wheel
        uniform_int_distribution<int> uniintdist(0, num_particles-1);
        int index = uniintdist(gen);	

        std::vector<Particle> resampled_particles;

        double beta = 0;
        double wmax = *max_element(weights.begin(), weights.end());

        uniform_real_distribution<double> unimaxWeights(0.0, wmax);

        for (int i = 0; i< num_particles; i++)
        {
             beta += 2.0 * unimaxWeights(gen);

             while (beta > weights[index])
             {
                beta -= weights[index];
                index = (index + 1) % num_particles;
             }
             resampled_particles.push_back(particles[index]);
        }

        particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
